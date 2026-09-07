# -*- coding: utf-8 -*-
"""Tests for the retrieval-aligned reranker (`src/train_reranker.py`).

The version ladder v1..v5 exists because the original ranker was measurably
worse than no ranking at all.  Two things must not silently break:

* negatives must come from the retrieval candidate pool, and with ``serve_k``
  they must reproduce the *serving* truncation exactly;
* the residual ranker must reproduce the retrieval order at initialisation,
  which is what bounds v5's worst case to "as good as no reranking".
"""
import unittest

import numpy as np
import torch

from train_reranker import CrossDeepFM, sample_groups, split_train_valid


def make_pool(n_users=3, depth=10):
    """pool[u] = [0..depth) 打乱前的自然顺序，便于断言截断行为。"""
    return np.tile(np.arange(depth, dtype=np.int64), (n_users, 1))


class SampleGroupsTests(unittest.TestCase):
    def setUp(self):
        import pandas as pd
        self.pairs = pd.DataFrame({"u": [0, 0], "i": [1, 2]})
        self.pool = make_pool()
        self.positives = {0: {1, 2, 3}}
        self.rng = np.random.default_rng(0)

    def test_positive_is_always_the_first_column(self):
        groups, users = sample_groups(
            self.pairs, self.pool, self.positives, self.rng, 3)
        self.assertEqual(groups[:, 0].tolist(), [1, 2])
        self.assertEqual(users.tolist(), [0, 0])

    def test_negatives_never_include_any_user_positive(self):
        groups, _ = sample_groups(
            self.pairs, self.pool, self.positives, self.rng, 4)
        for row in groups:
            self.assertFalse(set(row[1:].tolist()) & self.positives[0])

    def test_negatives_come_from_the_retrieval_pool(self):
        groups, _ = sample_groups(
            self.pairs, self.pool, self.positives, self.rng, 4)
        pool_items = set(self.pool[0].tolist())
        for row in groups:
            self.assertTrue(set(row[1:].tolist()) <= pool_items)

    def test_serve_k_truncates_to_the_served_candidate_set(self):
        """serve_k 必须复刻线上的"剔除已看 -> 取前 k"，而不是全池均匀采。

        pool = [0..9]，用户正样本 {1,2,3}，目标 i=1。线上会滤掉其它正样本
        {2,3}，剩下 [0,1,4,5,6,7,8,9]，取前 5 -> [0,1,4,5,6]，去掉目标后
        负样本只能来自 {0,4,5,6}。
        """
        import pandas as pd
        pairs = pd.DataFrame({"u": [0], "i": [1]})
        groups, _ = sample_groups(
            pairs, self.pool, self.positives, self.rng, 4, serve_k=5)
        self.assertEqual(len(groups), 1)
        self.assertEqual(set(groups[0, 1:].tolist()), {0, 4, 5, 6})

    def test_group_is_dropped_when_too_few_negatives_exist(self):
        import pandas as pd
        pairs = pd.DataFrame({"u": [0], "i": [1]})
        # serve_k=2 -> 候选 [0,1]，去掉目标只剩 1 个负样本，要 5 个则丢弃
        groups, users = sample_groups(
            pairs, self.pool, self.positives, self.rng, 5, serve_k=2)
        self.assertEqual(len(groups), 0)
        self.assertEqual(len(users), 0)


class SplitTrainValidTests(unittest.TestCase):
    def test_last_interaction_per_user_goes_to_validation(self):
        import pandas as pd
        frame = pd.DataFrame({
            "u": [0, 0, 0, 1, 1],
            "i": [10, 11, 12, 20, 21],
            "timestamp": [1, 3, 2, 5, 4]})
        train, valid = split_train_valid(frame)
        self.assertEqual(sorted(valid["i"].tolist()), [11, 20])
        self.assertEqual(sorted(train["i"].tolist()), [10, 12, 21])

    def test_train_and_valid_do_not_overlap(self):
        import pandas as pd
        frame = pd.DataFrame({
            "u": [0, 0, 1, 1], "i": [1, 2, 3, 4],
            "timestamp": [1, 2, 1, 2]})
        train, valid = split_train_valid(frame)
        self.assertFalse(set(train.index) & set(valid.index))
        self.assertEqual(len(train) + len(valid), len(frame))


class ResidualRankerTests(unittest.TestCase):
    """v5 的核心保证：初始化时输出与召回顺序一致，精排只能做增量。"""

    FIELDS = [5, 7, 2, 3, 4]

    def make_batch(self, n=6):
        x = torch.zeros(n, 5, dtype=torch.long)
        gid = torch.zeros(n, 2, dtype=torch.long)
        mask = torch.ones(n, 2)
        dense = torch.stack([
            torch.linspace(0.9, 0.1, n),          # tt_score，递减
            torch.rand(n)], dim=1)
        return x, gid, mask, dense

    def test_at_init_the_ranking_equals_the_retrieval_ranking(self):
        torch.manual_seed(0)
        model = CrossDeepFM(self.FIELDS, residual=True).eval()
        x, gid, mask, dense = self.make_batch()
        with torch.no_grad():
            scores = model(x, gid, mask, dense)
        # gate 初始为 0 -> 打分完全由 tt_score 决定，顺序必须一致
        self.assertEqual(torch.argsort(-scores).tolist(),
                         torch.argsort(-dense[:, 0]).tolist())

    def test_gate_starts_at_zero_and_retrieval_weight_is_positive(self):
        model = CrossDeepFM(self.FIELDS, residual=True)
        self.assertAlmostEqual(float(model.gate.item()), 0.0)
        self.assertGreater(float(model.retrieval_weight.item()), 0.0)

    def test_residual_model_requires_dense_features(self):
        model = CrossDeepFM(self.FIELDS, residual=True).eval()
        x, gid, mask, _ = self.make_batch()
        with self.assertRaises(ValueError):
            model(x, gid, mask, None)

    def test_non_residual_cross_head_is_zero_initialised(self):
        """v3：交叉特征接上的瞬间不应改变已有打分（等价于 v2）。"""
        torch.manual_seed(0)
        model = CrossDeepFM(self.FIELDS, residual=False).eval()
        x, gid, mask, dense = self.make_batch()
        with torch.no_grad():
            with_dense = model(x, gid, mask, dense)
            without = model(x, gid, mask, None)
        self.assertTrue(torch.allclose(with_dense, without, atol=1e-6))

    def test_learned_part_can_reorder_once_the_gate_opens(self):
        torch.manual_seed(0)
        model = CrossDeepFM(self.FIELDS, residual=True).eval()
        x, gid, mask, dense = self.make_batch()
        # 默认最后一层零初始化保证 gate=0 时不变序；本测试要证明 gate 打开后
        # 学习项确实能改变顺序，因此把 dense head 换成确定性的线性映射。
        model.dense = torch.nn.Linear(2, 1)
        with torch.no_grad():
            model.dense.weight.copy_(torch.tensor([[1.0, -2.0]]))
            model.dense.bias.zero_()
            model.gate.fill_(50.0)
            model.retrieval_weight.fill_(0.0)
            scores = model(x, gid, mask, dense)
        # 完全关掉召回项后，顺序不再等于 tt_score 顺序（否则 gate 是死的）
        self.assertNotEqual(torch.argsort(-scores).tolist(),
                            torch.argsort(-dense[:, 0]).tolist())


if __name__ == "__main__":
    unittest.main()
