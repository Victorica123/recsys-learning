# -*- coding: utf-8 -*-
"""Tests for user-level GAUC and the tie-aware AUC used by the DeepFM report.

`gauc_score` is what the project reports next to the headline AUC.  Global AUC
mixes all users into one ranking, so a model that merely separates "users who
rate generously" from "users who do not" scores well without ever ordering a
single user's candidates correctly.  These tests pin that distinction down with
a hand-built case where the two metrics must disagree.
"""
import unittest

import numpy as np

from train_deepfm import auc_score, gauc_score


class AucTieHandlingTests(unittest.TestCase):
    """并列分数必须用平均秩，否则 AUC 会随输入顺序漂移。"""

    def test_all_scores_tied_gives_exactly_one_half(self):
        # 全部并列 = 模型没有任何区分度，AUC 的定义值是 0.5。
        auc = auc_score(np.array([1, 0, 1, 0]), np.array([0.5] * 4))
        self.assertAlmostEqual(auc, 0.5, places=12)

    def test_ties_are_order_independent(self):
        labels = np.array([1, 0, 1, 0, 1, 0])
        scores = np.array([0.9, 0.9, 0.4, 0.4, 0.1, 0.1])
        baseline = auc_score(labels, scores)
        for permutation in ([5, 4, 3, 2, 1, 0], [2, 0, 4, 1, 5, 3]):
            index = np.array(permutation)
            self.assertAlmostEqual(
                auc_score(labels[index], scores[index]), baseline, places=12)

    def test_partial_ties_match_the_pairwise_definition(self):
        # 并列对按 0.5 计：1 个正样本 vs 2 个负样本，其中与一个负样本并列。
        labels = np.array([1, 0, 0])
        scores = np.array([0.7, 0.7, 0.1])
        self.assertAlmostEqual(auc_score(labels, scores), 0.75, places=12)

    def test_single_class_returns_nan_instead_of_dividing_by_zero(self):
        self.assertTrue(np.isnan(auc_score(np.array([1, 1]), np.array([.2, .8]))))
        self.assertTrue(np.isnan(auc_score(np.array([0, 0]), np.array([.2, .8]))))


class GaucScoreTests(unittest.TestCase):
    def test_perfect_within_user_ranking_scores_one(self):
        labels = np.array([1, 0, 1, 0])
        scores = np.array([0.9, 0.1, 0.8, 0.2])
        groups = np.array([7, 7, 9, 9])
        self.assertAlmostEqual(gauc_score(labels, scores, groups)["gauc"], 1.0)

    def test_gauc_exceeds_global_auc_when_users_have_score_offsets(self):
        """GAUC 存在的主要理由：全局 AUC 会低估「组内排得很好」的模型。

        两个用户各自排得完美（组内 AUC 都是 1），但打分尺度不同：用户 b 的
        负样本(0.8)比用户 a 的正样本(0.5)还高。全局 AUC 把两人混在一起排序，
        被这个跨用户的尺度差拖到 0.75；GAUC 只认组内顺序，正确地给出 1.0。
        线上体验取决于同一个人看到的候选顺序，所以 GAUC 更贴近真实。
        """
        labels = np.array([1, 0, 1, 0])
        scores = np.array([0.5, 0.4, 0.9, 0.8])
        groups = np.array(["a", "a", "b", "b"])
        self.assertAlmostEqual(auc_score(labels, scores), 0.75, places=12)
        self.assertAlmostEqual(gauc_score(labels, scores, groups)["gauc"], 1.0)

    def test_gauc_is_zero_when_every_user_is_ranked_backwards(self):
        """反方向：每个用户内部都排反了，GAUC=0，而全局 AUC 仍是正数。

        全局 AUC 的 0.25 全部来自跨用户的比较，与「这个用户该看哪部」无关。
        """
        labels = np.array([1, 0, 1, 0])
        scores = np.array([0.80, 0.90, 0.10, 0.20])
        groups = np.array(["a", "a", "b", "b"])
        gauc = gauc_score(labels, scores, groups)["gauc"]
        self.assertAlmostEqual(gauc, 0.0)
        self.assertAlmostEqual(auc_score(labels, scores), 0.25, places=12)
        self.assertGreater(auc_score(labels, scores), gauc)

    def test_string_group_ids_are_supported(self):
        # group_id 未必是整数下标（可能是 actor_id / hash），不能依赖减法。
        result = gauc_score(
            np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.2]),
            np.array(["user-a", "user-a", "user-b", "user-b"]))
        self.assertAlmostEqual(result["gauc"], 1.0)
        self.assertEqual(result["groups_used"], 2)

    def test_single_class_users_are_excluded_and_counted(self):
        labels = np.array([1, 0, 1, 1, 0, 0])
        scores = np.array([0.9, 0.1, 0.5, 0.6, 0.2, 0.3])
        groups = np.array(["ok", "ok", "allpos", "allpos", "allneg", "allneg"])
        result = gauc_score(labels, scores, groups)
        self.assertEqual(result["groups_used"], 1)
        self.assertEqual(result["groups_skipped_single_class"], 2)
        self.assertAlmostEqual(result["coverage"], 1 / 3)
        self.assertAlmostEqual(result["gauc"], 1.0)

    def test_weighting_is_by_group_size(self):
        # 4 个样本的组 AUC=1，2 个样本的组 AUC=0 -> 加权 = 4/(4+2)
        labels = np.array([1, 1, 0, 0, 1, 0])
        scores = np.array([0.9, 0.8, 0.2, 0.1, 0.1, 0.9])
        groups = np.array(["big", "big", "big", "big", "small", "small"])
        result = gauc_score(labels, scores, groups)
        self.assertAlmostEqual(result["gauc"], 4 / 6)
        self.assertEqual(result["groups_used"], 2)

    def test_group_order_does_not_matter(self):
        labels = np.array([1, 0, 1, 0, 0, 1])
        scores = np.array([0.9, 0.3, 0.4, 0.8, 0.2, 0.7])
        groups = np.array([2, 1, 2, 1, 3, 3])
        shuffled = np.array([4, 1, 0, 5, 3, 2])
        self.assertAlmostEqual(
            gauc_score(labels, scores, groups)["gauc"],
            gauc_score(labels[shuffled], scores[shuffled],
                       groups[shuffled])["gauc"],
            places=12)

    def test_all_groups_single_class_returns_nan_not_zero(self):
        result = gauc_score(
            np.array([1, 1, 0, 0]), np.array([.1, .2, .3, .4]),
            np.array(["a", "a", "b", "b"]))
        self.assertTrue(np.isnan(result["gauc"]))
        self.assertEqual(result["groups_used"], 0)
        self.assertEqual(result["coverage"], 0.0)


if __name__ == "__main__":
    unittest.main()
