import tempfile
import unittest
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

from enterprise_evaluation import (
    business_metrics, evaluate_dataset, history_popularity, ranking_metrics,
)
from enterprise_recommendation import KnowledgeValidationError, item_key, timestamp
from tests.enterprise_fixture import fixture


class EnterpriseEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, _, cls.dataset = fixture()

    def test_metrics_have_a_known_oracle_and_reject_unjudged_items(self):
        labels = {"a": 3, "b": 2, "c": 0}
        result = ranking_metrics(["a", "b", "c"], labels, 2)
        self.assertEqual(result["ndcg_at_2"], 1.0)
        self.assertEqual(result["mrr_at_2"], 1.0)
        self.assertEqual(result["recall_at_2"], 1.0)
        with self.assertRaises(KnowledgeValidationError):
            ranking_metrics(["unknown"], labels)

    def test_delayed_train_outcomes_and_validation_data_cannot_enter_features(self):
        request = next(r for r in self.dataset.requests if r["split"] == "test")
        candidate = request["candidates"][0]
        cutoff = timestamp(self.dataset.train_end)
        base = {
            **request, **{key: candidate[key] for key in ("item_type", "item_id", "item_version")},
            "event_type": "open", "split": "train",
        }
        early = {**base, "recommendation_id": "early",
                 "occurred_at": (cutoff - timedelta(seconds=1)).isoformat()}
        late = {**base, "recommendation_id": "late", "occurred_at": cutoff.isoformat()}
        validation = {**early, "recommendation_id": "val", "split": "validation"}
        modified = replace(self.dataset, events=[early, late, validation])
        self.assertEqual(history_popularity(modified, request), {item_key(candidate): 1.0})
        later = {**request, "requested_at": (timestamp(request["requested_at"]) +
                                           timedelta(days=10)).isoformat()}
        self.assertEqual(history_popularity(modified, request), history_popularity(modified, later))

    def test_paired_comparisons_share_all_requests_and_cannot_promote_synthetic_data(self):
        report, slates = evaluate_dataset(self.dataset)
        for split, policies in report["offline_paired"].items():
            for metrics in policies.values():
                self.assertEqual(metrics["requests"], self.dataset.manifest["split_counts"][split])
        self.assertEqual(len(slates), 3 * len(self.dataset.requests))
        self.assertEqual(report["source_audit"]["upstream_shared_actors_between_policies"], 0)
        self.assertLess(report["source_audit"]["mean_logged_pool_fraction_of_authorized_catalog"], 1)
        self.assertEqual(report["decision"]["status"], "not_promoted_synthetic_only")
        self.assertIsNone(report["decision"]["real_business_lift"])
        self.assertFalse(report["decision"]["train_two_tower"])
        self.assertTrue(any(report["off_policy_support"]["unsupported_target_positions"].values()))

    def test_business_outcomes_are_only_reported_for_logged_policy_cohorts(self):
        report = business_metrics(self.dataset)
        self.assertTrue(report["cohorts"])
        for name, metrics in report["cohorts"].items():
            self.assertIn(name.split("/")[1], {"recency_popularity_v1", "semantic_hybrid_v1"})
            for field, value in metrics.items():
                if "_per_" in field and value is not None:
                    self.assertGreaterEqual(value, 0)
                    self.assertLessEqual(value, 1)

    def test_complete_pipeline_exports_portable_artifacts_and_refuses_reused_tag(self):
        from scripts.enterprise_pilot import run
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = run(tag="test", requests=40, output_root=root)
            for name in ("report.json", "report.md", "replay.jsonl", "policy.json",
                         "manifest.json", "enterprise_recommendation.py", "agent_service_example.json"):
                self.assertTrue((output / name).is_file())
            before = (output / "report.json").read_bytes()
            with self.assertRaises(FileExistsError):
                run(tag="test", requests=40, output_root=root)
            self.assertEqual(before, (output / "report.json").read_bytes())
            with self.assertRaises(ValueError):
                run(tag="../escape", requests=40, output_root=root)


if __name__ == "__main__":
    unittest.main()
