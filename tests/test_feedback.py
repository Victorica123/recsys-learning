"""CPU-only tests for the production feedback loop and replay estimator."""
import json
import random
import tempfile
import unittest
from pathlib import Path

from feedback import (
    bootstrap_mean_interval,
    bootstrap_cluster_mean_interval,
    compare_logged_policies,
    diversity_rerank,
    FeedbackStore,
    FeedbackValidationError,
    evaluate_top_k_policy,
    select_slate,
    write_jsonl,
)


def candidates():
    return [
        {"movie_id": 10, "title": "A", "genres": "Drama", "score": 0.9},
        {"movie_id": 20, "title": "B", "genres": "Comedy", "score": 0.8},
    ]


class SlateSelectionTests(unittest.TestCase):
    def test_deterministic_policy_has_zero_support_outside_top_k(self):
        served, logged = select_slate(candidates(), 1, 0.0)
        self.assertEqual([row["movie_id"] for row in served], [10])
        self.assertEqual(
            [row["behavior_propensity"] for row in logged], [1.0, 0.0])

    def test_epsilon_mixture_logs_exact_marginal_propensities(self):
        served, logged = select_slate(
            candidates(), 1, 0.2, rng=random.Random(1))
        self.assertEqual(len(served), 1)
        self.assertAlmostEqual(logged[0]["behavior_propensity"], 0.9)
        self.assertAlmostEqual(logged[1]["behavior_propensity"], 0.1)


class FeedbackStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "events.sqlite3"
        self.store = FeedbackStore(self.path)

    def tearDown(self):
        self.temporary.cleanup()

    def _record(self, recommendation_id="rec-1", exploration=0.2):
        _, logged = select_slate(
            candidates(), 1, exploration, rng=random.Random(2))
        self.store.record_recommendation(
            recommendation_id=recommendation_id,
            request_id="request-1",
            user_id=1,
            model_version="test-v1",
            policy_name="epsilon_slate_v1",
            exploration_rate=exploration,
            requested_k=1,
            candidates=logged,
            created_at="2026-01-01T00:00:00+00:00",
        )
        return logged

    def test_idempotent_impression_and_feedback_round_trip(self):
        logged = self._record()
        served_id = next(
            row["movie_id"] for row in logged
            if row["served_rank"] is not None)
        first = self.store.record_impressions("rec-1", [served_id])
        second = self.store.record_impressions("rec-1", [served_id])
        self.assertEqual(first["inserted"], 1)
        self.assertEqual(second["duplicates"], 1)

        event = self.store.record_feedback(
            recommendation_id="rec-1", movie_id=served_id,
            event_type="click", event_id="event-1")
        duplicate = self.store.record_feedback(
            recommendation_id="rec-1", movie_id=served_id,
            event_type="click", event_id="event-1")
        self.assertTrue(event["inserted"])
        self.assertTrue(duplicate["duplicate"])

        rows = self.store.export_rows()
        self.assertEqual(len(rows), 2)
        observed = [row for row in rows if row["impressed"]]
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0]["reward"], 1.0)
        self.assertIsNotNone(observed[0]["title"])
        self.assertIsNotNone(observed[0]["genres"])
        self.assertEqual(self.store.stats()["observed_ctr"], 1.0)
        with self.assertRaises(FeedbackValidationError):
            self.store.record_feedback(
                recommendation_id="rec-1", movie_id=served_id,
                event_type="like", event_id="event-1")

    def test_feedback_requires_an_impression(self):
        logged = self._record()
        served_id = next(
            row["movie_id"] for row in logged
            if row["served_rank"] is not None)
        with self.assertRaises(FeedbackValidationError):
            self.store.record_feedback(
                recommendation_id="rec-1", movie_id=served_id,
                event_type="click")

    def test_impression_rejects_unserved_item(self):
        logged = self._record(exploration=0.0)
        unserved_id = next(
            row["movie_id"] for row in logged
            if row["served_rank"] is None)
        with self.assertRaises(FeedbackValidationError):
            self.store.record_impressions("rec-1", [unserved_id])


class ReplayEvaluationTests(unittest.TestCase):
    def test_diversity_reranker_trades_small_score_gap_for_new_genre(self):
        rows = [
            {"movie_id": 1, "candidate_rank": 1, "score": 0.90,
             "genres": "Drama"},
            {"movie_id": 2, "candidate_rank": 2, "score": 0.89,
             "genres": "Drama"},
            {"movie_id": 3, "candidate_rank": 3, "score": 0.88,
             "genres": "Comedy"},
        ]
        selected = diversity_rerank(rows, 2, diversity_weight=0.1)
        self.assertEqual([row["movie_id"] for row in selected], [1, 3])

    def test_bootstrap_interval_is_exact_for_constant_values(self):
        self.assertEqual(
            bootstrap_mean_interval([0.25] * 5, samples=100),
            [0.25, 0.25])
        self.assertEqual(
            bootstrap_cluster_mean_interval(
                [0.25, 0.25], ["actor-a", "actor-b"], samples=100),
            [0.25, 0.25])

    def test_formal_policy_gate_can_promote_supported_candidate(self):
        rows = []
        for recommendation_id in ("a", "b"):
            for rank, (movie_id, score, genres, reward) in enumerate([
                (1, 0.90, "Drama", 0.0),
                (2, 0.89, "Drama", 0.0),
                (3, 0.88, "Comedy", 1.0),
            ], start=1):
                rows.append({
                    "recommendation_id": recommendation_id,
                    "actor_id": f"actor-{recommendation_id}",
                    "movie_id": movie_id,
                    "candidate_rank": rank,
                    "served_rank": rank,
                    "score": score,
                    "genres": genres,
                    "behavior_propensity": 1.0,
                    "impressed": True,
                    "reward": reward,
                })
        report = compare_logged_policies(
            rows, target_k=2, diversity_weight=0.1,
            bootstrap_samples=100, min_recommendations=2,
            min_actors=2,
            min_effective_sample_size=1,
            min_impression_coverage=1.0)
        self.assertTrue(report["gate"]["formal_ready"])
        self.assertEqual(report["gate"]["decision"], "promote_candidate")
        self.assertGreater(
            report["paired_delta"]["confidence_interval"][0], 0.0)

    def test_formal_policy_gate_refuses_tiny_sample(self):
        rows = [{
            "recommendation_id": "a", "movie_id": 1,
            "candidate_rank": 1, "served_rank": 1, "score": 0.9,
            "genres": "Drama", "behavior_propensity": 1.0,
            "impressed": True, "reward": 1.0,
        }]
        report = compare_logged_policies(
            rows, target_k=1, bootstrap_samples=20,
            min_recommendations=200, min_effective_sample_size=100)
        self.assertFalse(report["gate"]["formal_ready"])
        self.assertEqual(report["gate"]["decision"], "collect_more_data")
        self.assertGreaterEqual(len(report["gate"]["blockers"]), 2)

    def test_ips_uses_unobserved_candidates_and_reports_support(self):
        rows = [
            {
                "recommendation_id": "a", "candidate_rank": 1,
                "behavior_propensity": 0.75, "impressed": True, "reward": 1.0,
            },
            {
                "recommendation_id": "a", "candidate_rank": 2,
                "behavior_propensity": 0.25, "impressed": False, "reward": None,
            },
            {
                "recommendation_id": "b", "candidate_rank": 1,
                "behavior_propensity": 0.75, "impressed": False, "reward": None,
            },
            {
                "recommendation_id": "b", "candidate_rank": 2,
                "behavior_propensity": 0.25, "impressed": True, "reward": 0.0,
            },
        ]
        result = evaluate_top_k_policy(rows, target_k=1)
        self.assertTrue(result["support_ok"])
        self.assertAlmostEqual(result["ips_reward_per_item"], 2 / 3)
        self.assertEqual(result["snips_reward_per_item"], 1.0)

    def test_zero_propensity_is_an_explicit_support_failure(self):
        rows = [{
            "recommendation_id": "a", "candidate_rank": 1,
            "behavior_propensity": 0.0, "impressed": False, "reward": None,
        }]
        result = evaluate_top_k_policy(rows, target_k=1)
        self.assertFalse(result["support_ok"])
        self.assertIsNone(result["ips_reward_per_item"])
        self.assertIn("zero-propensity", result["warning"])

    def test_jsonl_writer_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.jsonl"
            write_jsonl([{"x": 1}], path)
            self.assertEqual(json.loads(path.read_text()), {"x": 1})
            with self.assertRaises(FileExistsError):
                write_jsonl([{"x": 2}], path)


if __name__ == "__main__":
    unittest.main()
