from __future__ import annotations

import unittest

import numpy as np

from multistage_retrieval import (
    ListwiseGroup,
    ResidualListwiseRanker,
    fit_residual_listwise_ranker,
    reciprocal_rank_fusion,
    sequence_candidate_features,
    validate_prior_window_groups,
)


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_item_supported_by_two_routes_beats_single_route_item(self):
        items, scores = reciprocal_rank_fusion({
            "a": [1, 2, 3], "b": [3, 4, 5]}, rrf_constant=0)
        self.assertEqual(items[0], 3)
        self.assertEqual(len(items), len(scores))

    def test_ties_are_broken_by_item_id(self):
        items, _scores = reciprocal_rank_fusion({"a": [9], "b": [2]})
        self.assertEqual(items, [2, 9])


class SequenceFeatureTests(unittest.TestCase):
    def test_recent_genre_match_and_full_route_ranks_are_encoded(self):
        features = sequence_candidate_features(
            [2, 3],
            routes={"two_tower": [2, 3], "item_knn": [3, 2], "popular": [2, 3]},
            recent_items=[0, 1],
            item_genres=[{0}, {1}, {1}, {2}],
            item_counts=np.asarray([10, 5, 3, 1]),
            long_tail_items={3},
        )
        self.assertEqual(features.shape, (2, 7))
        self.assertGreater(features[0, 3], features[1, 3])
        self.assertEqual(features[0, 4], 1.0)
        self.assertEqual(features[1, 6], 1.0)


class ResidualListwiseRankerTests(unittest.TestCase):
    def make_group(self):
        return ListwiseGroup(
            candidate_ids=np.asarray([10, 20]),
            base_scores=np.asarray([1.0, 0.95], dtype=np.float32),
            features=np.asarray([[0.0], [1.0]], dtype=np.float32),
            target_index=1, user_id=1, event_timestamp=1, window=1)

    def test_zero_residual_exactly_preserves_base_order(self):
        ranker = ResidualListwiseRanker.zero(1)
        self.assertEqual(ranker.rank(self.make_group(), 2), [10, 20])

    def test_listwise_training_can_promote_consistent_target_feature(self):
        groups = [self.make_group() for _ in range(20)]
        ranker, summary = fit_residual_listwise_ranker(
            groups, epochs=80, lr=0.05, l2=0.0, residual_scale=0.2)
        self.assertEqual(ranker.rank(groups[0], 1), [20])
        self.assertLess(summary["final_loss"], summary["initial_loss"])

    def test_training_refuses_groups_without_retrieved_positive(self):
        group = self.make_group()
        group.target_index = None
        with self.assertRaisesRegex(ValueError, "no ranker groups"):
            fit_residual_listwise_ranker([group], epochs=1)

    def test_same_or_future_window_group_is_rejected(self):
        group = self.make_group()
        group.window = 2
        with self.assertRaisesRegex(ValueError, "ranker leakage"):
            validate_prior_window_groups([group], evaluation_window=2)
        validate_prior_window_groups([group], evaluation_window=3)


if __name__ == "__main__":
    unittest.main()
