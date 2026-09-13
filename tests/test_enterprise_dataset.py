import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from enterprise_dataset import EnterpriseDataset, validate_request
from enterprise_feedback import KnowledgeConflictError, KnowledgeFeedbackStore
from enterprise_recommendation import KnowledgeAuthorizationError, KnowledgeValidationError, Principal
from enterprise_seed import generate_dataset
from tests.enterprise_fixture import fixture


class EnterpriseDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw, cls.manifest, cls.dataset = fixture()

    def test_generator_reproduces_all_upstream_jsonl_hashes_and_refuses_overwrite(self):
        reference = json.loads((Path(__file__).resolve().parents[1] /
                                "artifacts/enterprise_seed_provenance.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "seed"
            manifest = generate_dataset(output)
            self.assertEqual(manifest["files"], reference["upstream_manifest"]["files"])
            EnterpriseDataset.load(output)
            with self.assertRaises(FileExistsError):
                generate_dataset(output)
            self.assertEqual(manifest, json.loads((output / "manifest.json").read_text(encoding="utf-8")))

    def test_manifest_detects_changed_input_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "seed"
            generate_dataset(output, requests=40)
            with (output / "items.jsonl").open("ab") as handle:
                handle.write(b"\n")
            with self.assertRaisesRegex(KnowledgeValidationError, "hash mismatch"):
                EnterpriseDataset.load(output)

    def test_cross_tenant_candidate_and_role_spoofing_are_rejected(self):
        raw = copy.deepcopy(self.raw)
        raw["recommendations.jsonl"][0]["candidates"][0]["item_id"] = "item_02_01_01"
        with self.assertRaises(KnowledgeValidationError):
            EnterpriseDataset.from_rows(raw, self.manifest)
        raw = copy.deepcopy(self.raw)
        raw["recommendations.jsonl"][0]["user_role"] = "operations"
        with self.assertRaises(KnowledgeValidationError):
            EnterpriseDataset.from_rows(raw, self.manifest)

    def test_split_is_checked_against_time_not_just_sort_order(self):
        raw = copy.deepcopy(self.raw)
        raw["recommendations.jsonl"][0]["split"] = "test"
        with self.assertRaisesRegex(KnowledgeValidationError, "actual timestamp"):
            EnterpriseDataset.from_rows(raw, self.manifest)

    def test_explicit_workspace_and_item_versions_are_not_silently_rewritten(self):
        raw = copy.deepcopy(self.raw)
        raw["recommendations.jsonl"][0]["workspace_id"] = "another-workspace"
        with self.assertRaisesRegex(KnowledgeValidationError, "workspace"):
            EnterpriseDataset.from_rows(raw, self.manifest)
        raw = copy.deepcopy(self.raw)
        raw["recommendations.jsonl"][0]["candidates"][0]["item_version"] = 999
        with self.assertRaisesRegex(KnowledgeValidationError, "type/version"):
            EnterpriseDataset.from_rows(raw, self.manifest)

    def test_duplicate_candidates_and_false_unserved_propensities_are_rejected(self):
        request = copy.deepcopy(self.dataset.requests[0])
        request["candidates"][1] = copy.deepcopy(request["candidates"][0])
        with self.assertRaises(KnowledgeValidationError):
            validate_request(request)
        request = copy.deepcopy(self.dataset.requests[1])
        unserved = next(row for row in request["candidates"] if row["served_rank"] is None)
        unserved["behavior_propensity"] = 0.1
        with self.assertRaisesRegex(KnowledgeValidationError, "epsilon-slate"):
            validate_request(request)

    def test_event_file_order_is_not_assumed_to_be_time_order(self):
        raw = copy.deepcopy(self.raw)
        raw["events.jsonl"].reverse()
        actual = EnterpriseDataset.from_rows(raw, self.manifest)
        self.assertEqual(len(actual.events), len(self.dataset.events))

    def test_feedback_timestamp_and_action_lineage_are_enforced(self):
        raw = copy.deepcopy(self.raw)
        event = next(row for row in raw["events.jsonl"] if row["event_type"] == "open")
        request = next(row for row in raw["recommendations.jsonl"]
                       if row["recommendation_id"] == event["recommendation_id"])
        event["occurred_at"] = request["requested_at"]
        with self.assertRaises(KnowledgeValidationError):
            EnterpriseDataset.from_rows(raw, self.manifest)
        raw = copy.deepcopy(self.raw)
        event = next(row for row in raw["events.jsonl"] if row["event_type"] == "complete_action")
        event["metadata"]["action_id"] = "unknown-action"
        with self.assertRaisesRegex(KnowledgeValidationError, "matching creation"):
            EnterpriseDataset.from_rows(raw, self.manifest)

    def test_incomplete_oracle_labels_are_rejected(self):
        raw = copy.deepcopy(self.raw)
        raw["ground_truth.jsonl"][0]["labels"].pop()
        with self.assertRaisesRegex(KnowledgeValidationError, "exactly the logged pool"):
            EnterpriseDataset.from_rows(raw, self.manifest)


class KnowledgeFeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, _, cls.dataset = fixture()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = KnowledgeFeedbackStore(Path(self.directory.name) / "feedback.sqlite3")
        self.request = self.dataset.requests[0]
        self.principal = self.dataset.principal(self.request)

    def tearDown(self):
        self.store.__exit__()
        self.directory.cleanup()

    def test_atomic_import_is_idempotent_and_scope_exports_are_separate(self):
        first = self.store.import_dataset(self.dataset)
        self.assertEqual(first["requests_added"], len(self.dataset.requests))
        self.assertEqual(first["events_added"], len(self.dataset.events))
        counts = self.store.counts()
        self.assertEqual(self.store.import_dataset(self.dataset),
                         {"requests_added": 0, "events_added": 0})
        self.assertEqual(counts, self.store.counts())
        rows = list(self.store.replay(self.principal.tenant_id, self.principal.workspace_id))
        self.assertTrue(rows)
        self.assertTrue(all(row["recommendation"]["tenant_id"] == self.principal.tenant_id
                            for row in rows))
        self.assertEqual(list(self.store.replay("unknown", self.principal.workspace_id)), [])

    def test_same_id_with_changed_payload_conflicts(self):
        self.store.record_request(self.request, self.principal)
        changed = {**self.request, "model_version": "different-model"}
        with self.assertRaises(KnowledgeConflictError):
            self.store.record_request(changed, self.principal)
        event = self.dataset.events[0]
        self.store.record_event(event, self.principal)
        self.assertFalse(self.store.record_event(event, self.principal))
        with self.assertRaises(KnowledgeConflictError):
            self.store.record_event({**event, "value": 2.0}, self.principal)
        with self.assertRaises(KnowledgeConflictError):
            self.store.record_event({**event, "event_id": "another-impression-id"}, self.principal)

    def test_cross_actor_and_unserved_feedback_are_rejected(self):
        self.store.record_request(self.request, self.principal)
        other = Principal(self.principal.tenant_id, self.principal.workspace_id,
                          "mallory", self.principal.role)
        event = {**self.dataset.events[0], "actor_id": "mallory"}
        with self.assertRaises(KnowledgeAuthorizationError):
            self.store.record_event(event, other)
        bad_version = {**self.dataset.events[0], "item_version": 999}
        with self.assertRaises(KnowledgeValidationError):
            self.store.record_event(bad_version, self.principal)
        self.assertEqual(self.store.counts()["events"], 0)

    def test_failed_batch_rolls_back_requests_and_events(self):
        events = copy.deepcopy(self.dataset.events)
        events[-1]["item_version"] = 999
        with self.assertRaises(KnowledgeValidationError):
            self.store.import_dataset(replace(self.dataset, events=events))
        self.assertEqual(self.store.counts(), {"requests": 0, "candidates": 0, "events": 0})

    def test_actor_deletion_cascades_without_touching_other_tenants(self):
        self.store.import_dataset(self.dataset)
        other = next(tenant for tenant in self.dataset.tenants.values()
                     if tenant["tenant_id"] != self.principal.tenant_id)
        previous = list(self.store.replay(other["tenant_id"], other["workspace_id"]))
        deleted = self.store.delete_actor(self.principal)
        self.assertGreater(deleted, 0)
        self.assertEqual(previous, list(self.store.replay(other["tenant_id"], other["workspace_id"])))
        self.assertFalse(any(row["recommendation"]["actor_id"] == self.principal.actor_id
                             for row in self.store.replay(
                                 self.principal.tenant_id, self.principal.workspace_id)))
        self.assertEqual(self.store.db.execute("PRAGMA foreign_key_check").fetchall(), [])


if __name__ == "__main__":
    unittest.main()
