# -*- coding: utf-8 -*-
"""Tests for the end-to-end retrieval->ranking evaluation helpers.

The headline finding these helpers produce (DeepFM reranking *lowers*
end-to-end Recall@10 from 0.098 to 0.065 and halves catalog coverage) is only
credible if the funnel bookkeeping itself is right, so the pure helpers are
pinned here.  The full pipeline run needs real checkpoints and stays out of the
default CPU smoke suite.
"""
import unittest
from collections import Counter

from scripts.eval_end_to_end import dcg_hit, exposure_profile, rank_of


class RankOfTests(unittest.TestCase):
    def test_returns_zero_based_position_inside_cutoff(self):
        self.assertEqual(rank_of([5, 9, 3], 9, 3), 1)
        self.assertEqual(rank_of([5, 9, 3], 5, 3), 0)

    def test_returns_none_when_beyond_cutoff(self):
        # 物品存在但排在 cutoff 之外 —— 对 Recall@K 而言等同于没命中。
        self.assertIsNone(rank_of([5, 9, 3], 3, 2))

    def test_returns_none_when_absent(self):
        self.assertIsNone(rank_of([5, 9, 3], 42, 3))

    def test_empty_list_is_a_miss_not_a_crash(self):
        self.assertIsNone(rank_of([], 1, 10))


class DcgHitTests(unittest.TestCase):
    def test_first_position_scores_one(self):
        self.assertAlmostEqual(dcg_hit(0), 1.0)

    def test_gain_decays_monotonically_with_position(self):
        values = [dcg_hit(position) for position in range(10)]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertAlmostEqual(dcg_hit(1), 1 / 1.5849625007211562, places=9)


class ExposureProfileTests(unittest.TestCase):
    def test_counts_distinct_items_and_head_share(self):
        # 10 个用户 x Top-2；物品 1 每次都出现，物品 2..11 各出现一次。
        counter = Counter({1: 10})
        counter.update({item: 1 for item in range(2, 12)})
        profile = exposure_profile(counter, k=2, top_n=1)
        self.assertEqual(profile["distinct_items"], 11)
        self.assertAlmostEqual(profile["top1_share"], 10 / 20)

    def test_uniform_exposure_has_low_head_share(self):
        counter = Counter({item: 1 for item in range(100)})
        profile = exposure_profile(counter, k=10, top_n=20)
        self.assertEqual(profile["distinct_items"], 100)
        self.assertAlmostEqual(profile["top20_share"], 0.2)

    def test_user_share_is_fraction_of_users_not_impressions(self):
        # 20 次曝光 / k=2 => 10 个用户；物品 1 出现在其中 10 个用户里。
        counter = Counter({1: 10})
        counter.update({item: 1 for item in range(2, 12)})
        profile = exposure_profile(counter, k=2)
        self.assertAlmostEqual(profile["most_exposed"][0]["user_share"], 1.0)
        self.assertEqual(profile["most_exposed"][0]["movie_id"], 1)

    def test_empty_counter_returns_zero_without_dividing(self):
        profile = exposure_profile(Counter(), k=10)
        self.assertEqual(profile["distinct_items"], 0)
        self.assertIsNone(profile["top20_share"])


if __name__ == "__main__":
    unittest.main()
