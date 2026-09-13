import copy
import json
import math
import random
import tempfile
import unittest
from pathlib import Path

from enterprise_dataset import validate_request
from enterprise_recommendation import (
    KnowledgeAuthorizationError, KnowledgeRanker, KnowledgeValidationError,
    Principal, ScorerUnavailable, experiment_bucket, policy_bundle, timestamp,
)


class KnowledgeRankingTests(unittest.TestCase):
    def setUp(self):
        self.principal = Principal("tenant-a", "workspace-a", "alice", "backend_engineer")
        self.request = {
            "recommendation_id": "rec-1", "tenant_id": "tenant-a", "workspace_id": "workspace-a",
            "workspace_type": "team", "actor_id": "alice", "requested_at": "2026-02-01T09:00:00Z",
            "context_type": "analysis_result", "context_id": "analysis-1",
            "context": {"intent": "分片上传 checksum upload"}, "synthetic": True,
        }
        self.items = [
            self.item("upload", "分片上传 checksum upload", "knowledge"),
            self.item("billing", "账单配额 billing quota", "prd"),
            self.item("audit", "审计留存 audit retention", "incident"),
        ]

    def item(self, identity, title, kind):
        return {
            "tenant_id": "tenant-a", "workspace_id": "workspace-a", "workspace_type": "team",
            "owner_id": "alice", "visibility": "workspace", "visible_roles": ["backend_engineer"],
            "item_type": kind, "item_id": identity, "item_version": 1, "title": title,
            "summary": title, "content": title, "tags": [], "created_at": "2026-01-01T00:00:00Z",
            "status": "ACTIVE", "revoked_at": None, "superseded_at": None,
        }

    def test_relevant_text_wins_without_oracle_fields(self):
        result = KnowledgeRanker().recommend(self.principal, self.request, self.items,
                                             policy="semantic_v1")
        self.assertEqual(result["candidates"][0]["item_id"], "upload")
        self.assertEqual(result["model_version"], "knowledge-bm25-v1")

    def test_authorization_precedes_text_and_corpus_statistics(self):
        ranker = KnowledgeRanker()
        expected = ranker.recommend(self.principal, self.request, self.items)
        forbidden = []
        for field, value in [("tenant_id", "tenant-b"), ("workspace_id", "workspace-b"),
                             ("visible_roles", ["operations"]), ("status", "DRAFT"),
                             ("status", "REVOKED"), ("created_at", "2027-01-01T00:00:00Z")]:
            item = {**self.items[0], "item_id": "secret-" + field, field: value, "content": object()}
            forbidden.append(item)
        actual = ranker.recommend(self.principal, self.request, self.items + forbidden)
        self.assertEqual(actual, expected)

    def test_scope_spoofing_is_rejected(self):
        for field, value in [("tenant_id", "tenant-b"), ("workspace_id", "workspace-b"),
                             ("actor_id", "mallory"), ("workspace_type", "personal")]:
            with self.subTest(field=field), self.assertRaises(KnowledgeAuthorizationError):
                KnowledgeRanker().recommend(self.principal, {**self.request, field: value}, self.items)

    def test_personal_workspace_enforces_owner(self):
        principal = Principal("tenant-a", "workspace-a", "alice", "backend_engineer", "personal")
        items = [{**item, "workspace_type": "personal"} for item in self.items]
        items[1]["owner_id"] = "bob"
        result = KnowledgeRanker().recommend(
            principal, {**self.request, "workspace_type": "personal"}, items)
        self.assertNotIn("billing", [item["item_id"] for item in result["candidates"]])

    def test_version_validity_at_the_exact_supersession_boundary(self):
        old = {**self.items[0], "status": "SUPERSEDED", "superseded_at": "2026-02-01T09:00:00Z"}
        new = {**self.items[0], "item_version": 2, "created_at": "2026-02-01T09:00:00Z"}
        ranker = KnowledgeRanker()
        before = ranker.recommend(
            self.principal, {**self.request, "requested_at": "2026-02-01T08:59:59Z"}, [old, new])
        at = ranker.recommend(self.principal, self.request, [old, new, new])
        self.assertEqual([row["item_version"] for row in before["candidates"]], [1])
        self.assertEqual([row["item_version"] for row in at["candidates"]], [2])
        revoked = {**new, "status": "REVOKED", "revoked_at": self.request["requested_at"]}
        self.assertEqual(ranker.recommend(self.principal, self.request, [revoked])["candidates"], [])

    def test_oracle_and_logged_feature_poisoning_cannot_change_output(self):
        ranker = KnowledgeRanker()
        expected = ranker.recommend(self.principal, self.request, self.items)
        items = [{**item, "topic": "spoof", "relevance_grade": 999,
                  "popularity_prior": 1e100, "score": 1e100,
                  "features": {"topic_match": 1e100}} for item in reversed(self.items)]
        request = copy.deepcopy(self.request)
        request["context"]["topic"] = "billing"
        request["preferred_topics"] = ["billing"]
        request["candidates"] = [{"item_id": "billing", "score": 1e100}]
        self.assertEqual(expected, ranker.recommend(self.principal, request, items))

    def test_fallback_keeps_the_authorized_baseline_and_records_actual_policy(self):
        def unavailable(query, documents):
            self.assertEqual(len(documents), len(self.items))
            raise ScorerUnavailable("offline")
        actual = KnowledgeRanker(content_scorer=unavailable).recommend(
            self.principal, self.request, self.items)
        expected = KnowledgeRanker().recommend(
            self.principal, self.request, self.items, policy="baseline_v0")
        self.assertEqual(actual["policy_version"], "baseline_v0")
        self.assertEqual(actual["requested_policy_version"], "hybrid_v1")
        self.assertEqual(actual["fallback_reason"], "content_scorer_unavailable")
        self.assertEqual(actual["candidates"], expected["candidates"])

    def test_nonfinite_scorer_output_falls_back(self):
        result = KnowledgeRanker(content_scorer=lambda q, d: [math.nan] * len(d)).recommend(
            self.principal, self.request, self.items)
        self.assertEqual(result["policy_version"], "baseline_v0")

    def test_epsilon_inclusion_probabilities_and_short_slates(self):
        result = KnowledgeRanker().recommend(
            self.principal, self.request, self.items, k=2, epsilon=0.3, rng=random.Random(1))
        result["split"] = "train"
        validate_request(result)
        probabilities = [row["behavior_propensity"] for row in result["candidates"]]
        self.assertAlmostEqual(sum(probabilities), 2.0)
        self.assertAlmostEqual(probabilities[0], 0.7 + 0.3 * 2 / 3)
        self.assertAlmostEqual(probabilities[-1], 0.3 * 2 / 3)
        empty = KnowledgeRanker().recommend(self.principal, self.request, [])
        empty["split"] = "train"
        validate_request(empty)
        self.assertEqual(empty["candidate_count"], 0)

    def test_policy_artifact_round_trip_and_invalid_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            path.write_text(json.dumps(policy_bundle()), encoding="utf-8")
            actual = KnowledgeRanker.from_artifact(path).recommend(
                self.principal, self.request, self.items)
            self.assertEqual(actual, KnowledgeRanker().recommend(
                self.principal, self.request, self.items))
        bad = policy_bundle()
        bad["weights"]["hybrid_v1"]["content"] = -1
        with self.assertRaises(KnowledgeValidationError):
            KnowledgeRanker(bad)

    def test_stable_experiment_buckets_include_workspace_and_tenant(self):
        first = experiment_bucket(self.principal, "pilot-v1", buckets=2 ** 64)
        self.assertEqual(first, experiment_bucket(self.principal, "pilot-v1", buckets=2 ** 64))
        other = Principal("tenant-b", "workspace-a", "alice", "backend_engineer")
        self.assertNotEqual(first, experiment_bucket(other, "pilot-v1", buckets=2 ** 64))
        with self.assertRaises(KnowledgeValidationError):
            timestamp("2026-01-01T00:00:00")


if __name__ == "__main__":
    unittest.main()
