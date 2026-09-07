from __future__ import annotations

import unittest

import pandas as pd

from scripts.benchmark_retrieval import (
    aggregate_seed_runs,
    evaluate_recommendations,
    item_knn_recommendations,
    popular_recommendations,
    rolling_time_splits,
    summarize_across_windows,
)


def ratings_frame(rows):
    return pd.DataFrame(
        rows, columns=["u", "i", "rating", "timestamp"])


class RollingTimeSplitTests(unittest.TestCase):
    def setUp(self):
        self.ratings = ratings_frame([
            (0, 0, 5, 1), (1, 1, 5, 2), (0, 2, 5, 3), (1, 3, 5, 4),
            (0, 4, 5, 5), (1, 5, 5, 6), (0, 6, 5, 7), (1, 7, 5, 8),
            (0, 8, 5, 9), (1, 9, 5, 10),
        ])

    def test_every_window_has_strict_global_time_order(self):
        windows = rolling_time_splits(
            self.ratings, self.ratings,
            train_fracs=(0.4, 0.6), horizon_frac=0.2)
        for window in windows:
            self.assertLess(
                window["train"]["timestamp"].max(),
                window["test"]["timestamp"].min())
            self.assertLessEqual(
                window["test"]["timestamp"].max(),
                window["horizon_end_quantile_timestamp"])

    def test_first_future_positive_per_user_is_the_target(self):
        window = rolling_time_splits(
            self.ratings, self.ratings,
            train_fracs=(0.4,), horizon_frac=0.4)[0]
        self.assertEqual(window["test"].groupby("u").size().max(), 1)
        self.assertEqual(
            window["test"].set_index("u")["timestamp"].to_dict(),
            {0: 5, 1: 6})

    def test_invalid_or_overflowing_window_is_rejected(self):
        for fractions, horizon in [((0.9,), 0.2), ((), 0.1), ((0.5, 0.5), 0.1)]:
            with self.assertRaises(ValueError):
                rolling_time_splits(
                    self.ratings, self.ratings,
                    train_fracs=fractions, horizon_frac=horizon)


class FullCatalogMetricTests(unittest.TestCase):
    def test_popular_baseline_filters_seen_items(self):
        train = ratings_frame([
            (0, 0, 5, 1), (1, 0, 5, 2), (0, 1, 5, 3), (2, 2, 5, 4)])
        recs = popular_recommendations(train, [0], n_items=4, k=2)
        self.assertNotIn(0, recs[0])
        self.assertNotIn(1, recs[0])
        self.assertEqual(len(recs[0]), 2)

    def test_exact_metrics_include_distribution_and_full_catalog_contract(self):
        train = ratings_frame([
            (0, 0, 5, 1), (1, 0, 5, 2), (0, 1, 5, 3), (1, 2, 5, 4)])
        test = ratings_frame([(0, 2, 5, 5), (1, 3, 5, 6)])
        recs = {0: [2, 3], 1: [1, 4]}
        result = evaluate_recommendations(
            recs, train=train, test=test, n_items=5, k=2,
            bootstrap_samples=100)
        self.assertEqual(result["recall@2"], 0.5)
        self.assertEqual(result["ndcg@2"], 0.5)
        self.assertEqual(result["mrr@2"], 0.5)
        self.assertEqual(result["catalog_coverage@2"], 0.8)
        self.assertEqual(result["candidate_catalog_size"], 5)
        self.assertFalse(result["sampled_negatives"])

    def test_seen_item_is_a_hard_protocol_failure(self):
        train = ratings_frame([(0, 0, 5, 1)])
        test = ratings_frame([(0, 2, 5, 2)])
        with self.assertRaisesRegex(ValueError, "already-seen"):
            evaluate_recommendations(
                {0: [0, 2]}, train=train, test=test, n_items=3, k=2,
                bootstrap_samples=20)

    def test_duplicate_or_short_slate_is_rejected(self):
        train = ratings_frame([(0, 0, 5, 1)])
        test = ratings_frame([(0, 2, 5, 2)])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluate_recommendations(
                {0: [1, 1]}, train=train, test=test, n_items=3, k=2,
                bootstrap_samples=20)
        with self.assertRaisesRegex(ValueError, "expected 2"):
            evaluate_recommendations(
                {0: [1]}, train=train, test=test, n_items=3, k=2,
                bootstrap_samples=20)

    def test_item_knn_recent_context_still_filters_complete_history(self):
        train = ratings_frame([
            (0, 0, 5, 1), (1, 0, 5, 2),
            (0, 1, 5, 3), (1, 2, 5, 4),
            (0, 2, 5, 5), (1, 3, 5, 6),
        ])
        recs = item_knn_recommendations(
            train, [0], n_users=2, n_items=5, k=2,
            neighbors=2, history_limit=1)
        self.assertTrue(set(recs[0]).isdisjoint({0, 1, 2}))


class SummaryTests(unittest.TestCase):
    def test_seed_mean_precedes_cross_window_macro_average(self):
        rows = [
            {"window": 1, "model": "m", "recall@10": 0.0},
            {"window": 1, "model": "m", "recall@10": 1.0},
            {"window": 2, "model": "m", "recall@10": 0.5},
        ]
        seed_summary = aggregate_seed_runs(rows, "recall@10")
        self.assertEqual(seed_summary[0]["mean_recall@10"], 0.5)
        cross = summarize_across_windows(rows, "recall@10")[0]
        self.assertEqual(cross["macro_mean_recall@10"], 0.5)
        self.assertEqual(cross["windows"], 2)


if __name__ == "__main__":
    unittest.main()
