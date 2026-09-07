# -*- coding: utf-8 -*-
"""Tests for the two evaluation splits and their leakage properties.

`split_by_time` exists because leave-one-out lets the model see interactions
that happen *after* a user's test moment (time travel), which inflates Recall.
Measured on ML-1M: 0.0980 [0.0884, 0.1075] under leave-one-out versus
0.0691 [0.0553, 0.0847] under a global time split, intervals disjoint.

These tests pin the property that makes the time split trustworthy: **no
training interaction may be newer than any test interaction.**
"""
import unittest

import pandas as pd

from train_two_tower import split_by_time, split_leave_one_out


def frame(rows):
    """rows: (u, i, rating, timestamp)"""
    return pd.DataFrame(rows, columns=["u", "i", "rating", "timestamp"])


class SplitByTimeTests(unittest.TestCase):
    def setUp(self):
        # 10 条交互，时间 1..10，两个用户交错
        self.ratings = frame([
            (0, 10, 5, 1), (1, 20, 5, 2), (0, 11, 5, 3), (1, 21, 5, 4),
            (0, 12, 5, 5), (1, 22, 5, 6), (0, 13, 5, 7), (1, 23, 5, 8),
            (0, 14, 5, 9), (1, 24, 5, 10),
        ])
        self.pos = self.ratings[self.ratings["rating"] >= 4]

    def test_no_training_interaction_is_newer_than_any_test_interaction(self):
        """这是时间切分存在的唯一理由——没有时间穿越。"""
        train, test, _cold = split_by_time(self.ratings, self.pos, 0.3)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(test), 0)
        self.assertLess(train["timestamp"].max(), test["timestamp"].min())

    def test_leave_one_out_does_leak_across_time(self):
        """对照：留一法下，训练集里存在比某些测试时刻更晚的交互。

        用户 0 全程活跃在 t=1..3，用户 1 活跃在 t=5..7。留一法把用户 0 的
        t=3 当测试，而用户 1 的 t=5、t=6 仍留在训练集里——模型等于拿 t=5/6
        的信息去预测 t=3 发生的事。这正是全局时间切分要消除的东西。
        """
        ratings = frame([
            (0, 10, 5, 1), (0, 11, 5, 2), (0, 12, 5, 3),
            (1, 20, 5, 5), (1, 21, 5, 6), (1, 22, 5, 7),
        ])
        pos = ratings[ratings["rating"] >= 4]
        train, test, _cold = split_leave_one_out(ratings, pos)
        self.assertEqual(sorted(test["timestamp"].tolist()), [3, 7])
        # 训练集里有 t=5、6，晚于用户 0 的测试时刻 t=3 -> 时间穿越
        self.assertGreater(train["timestamp"].max(), test["timestamp"].min())

        # 同一份数据走时间切分：训练集不含任何晚于测试的交互
        t_train, t_test, _c = split_by_time(ratings, pos, 0.4)
        self.assertLess(t_train["timestamp"].max(), t_test["timestamp"].min())

    def test_test_set_takes_the_first_positive_after_the_cutoff(self):
        # test_frac=0.3 -> 切点在 t=7；用户 0 的测试期首个正样本是 t=9
        _train, test, _cold = split_by_time(self.ratings, self.pos, 0.3)
        first_by_user = test.groupby("u")["timestamp"].first().to_dict()
        self.assertEqual(first_by_user[0], 9)
        self.assertEqual(first_by_user[1], 8)

    def test_one_test_row_per_user(self):
        _train, test, _cold = split_by_time(self.ratings, self.pos, 0.5)
        self.assertEqual(len(test), test["u"].nunique())

    def test_cold_start_users_are_excluded_and_counted(self):
        """只在测试期出现的用户 embedding 从未训练过，必须剔除并计数。"""
        ratings = frame([
            (0, 10, 5, 1), (0, 11, 5, 2),
            (9, 30, 5, 9),                      # 用户 9 只出现在测试期
        ])
        pos = ratings[ratings["rating"] >= 4]
        train, test, cold = split_by_time(ratings, pos, 0.4)
        self.assertEqual(cold, 1)
        self.assertNotIn(9, set(test["u"]))
        self.assertNotIn(9, set(train["u"]))

    def test_negative_ratings_never_enter_either_side(self):
        ratings = frame([
            (0, 10, 5, 1), (0, 11, 2, 2), (0, 12, 5, 3), (0, 13, 1, 4),
            (1, 20, 5, 5), (1, 21, 5, 6),
        ])
        pos = ratings[ratings["rating"] >= 4]
        train, test, _cold = split_by_time(ratings, pos, 0.4)
        for part in (train, test):
            self.assertTrue((part["rating"] >= 4).all())

    def test_invalid_fraction_is_rejected(self):
        for bad in (0.0, 1.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                split_by_time(self.ratings, self.pos, bad)

    def test_larger_test_fraction_gives_a_smaller_training_set(self):
        small, _t1, _c1 = split_by_time(self.ratings, self.pos, 0.2)
        large, _t2, _c2 = split_by_time(self.ratings, self.pos, 0.6)
        self.assertGreater(len(small), len(large))


class LeaveOneOutTests(unittest.TestCase):
    def test_holds_out_exactly_the_last_rating_per_user(self):
        ratings = frame([
            (0, 10, 5, 1), (0, 11, 5, 5), (0, 12, 5, 3),
            (1, 20, 5, 2), (1, 21, 5, 4),
        ])
        pos = ratings[ratings["rating"] >= 4]
        train, test, cold = split_leave_one_out(ratings, pos)
        self.assertEqual(cold, 0)
        self.assertEqual(sorted(test["i"].tolist()), [11, 21])
        self.assertEqual(sorted(train["i"].tolist()), [10, 12, 20])

    def test_user_whose_last_rating_is_low_is_dropped_from_test(self):
        # 最后一条是 2 分 -> 该用户不进测试集（预测"下一部喜欢的"）
        ratings = frame([(0, 10, 5, 1), (0, 11, 2, 9)])
        pos = ratings[ratings["rating"] >= 4]
        _train, test, _cold = split_leave_one_out(ratings, pos)
        self.assertEqual(len(test), 0)

    def test_train_and_test_are_disjoint(self):
        ratings = frame([
            (0, 10, 5, 1), (0, 11, 5, 2), (1, 20, 4, 3), (1, 21, 5, 4)])
        pos = ratings[ratings["rating"] >= 4]
        train, test, _cold = split_leave_one_out(ratings, pos)
        self.assertFalse(set(train.index) & set(test.index))


if __name__ == "__main__":
    unittest.main()
