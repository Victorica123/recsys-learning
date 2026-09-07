# -*- coding: utf-8 -*-
"""Tests for the reusable serving core (`src/serving.py`).

The pure helpers (`pad_genre_batch`, `check_artifacts`) run everywhere with no
model load. The end-to-end recommender test loads real checkpoints, so it is
opt-in via ``RECSYS_INTEGRATION=1`` to keep the default smoke suite instant.
"""
import os
import time
import unittest

import torch

from serving import (Recommender, UnknownUserError, check_artifacts,
                     pad_genre_batch, validate_ranking_policy)


class PadGenreBatchTests(unittest.TestCase):
    def test_pads_ragged_lists_and_builds_mask(self):
        gid_b, mask_b = pad_genre_batch([[3], [3, 7, 1]])
        self.assertEqual(tuple(gid_b.shape), (2, 3))
        self.assertEqual(gid_b[0].tolist(), [3, 0, 0])       # right-padded
        self.assertEqual(mask_b[0].tolist(), [1.0, 0.0, 0.0])  # only real slots
        self.assertEqual(mask_b[1].tolist(), [1.0, 1.0, 1.0])

    def test_mask_selects_exactly_the_real_genres(self):
        _, mask_b = pad_genre_batch([[1, 2], [5], [4, 4, 4]])
        self.assertEqual(mask_b.sum(dim=1).tolist(), [2.0, 1.0, 3.0])

    def test_empty_batch_returns_placeholder_shapes(self):
        gid_b, mask_b = pad_genre_batch([])
        self.assertEqual(tuple(gid_b.shape), (0, 1))
        self.assertEqual(tuple(mask_b.shape), (0, 1))


class CheckArtifactsTests(unittest.TestCase):
    def test_returns_list(self):
        # In a complete checkout this is empty; the contract is "list of triples".
        missing = check_artifacts()
        self.assertIsInstance(missing, list)
        for entry in missing:
            self.assertEqual(len(entry), 3)

    def test_invalid_ranking_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_ranking_policy("auto")

    def test_ranker_checkpoint_requires_explicit_deepfm_policy(self):
        with self.assertRaises(ValueError):
            Recommender.load(deepfm_ckpt="checkpoints/deepfm.pt")


@unittest.skipUnless(os.environ.get("RECSYS_INTEGRATION") == "1",
                     "set RECSYS_INTEGRATION=1 to load real checkpoints")
class RecommenderIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rec = Recommender.load()
        cls.deepfm_rec = Recommender.load(ranking_policy="deepfm")

    def test_recommend_returns_k_valid_unseen_items(self):
        k = 10
        t0 = time.perf_counter()
        out = self.rec.recommend(user_id=1, k=k)
        latency_ms = (time.perf_counter() - t0) * 1000
        recs = out["recommendations"]
        self.assertLessEqual(len(recs), k)
        self.assertGreater(len(recs), 0)
        self.assertEqual(out["ranking_policy"], "retrieval")
        self.assertEqual(out["score_type"], "two_tower_similarity")
        self.assertIsNone(self.rec.deepfm)
        # scores are normalized-vector similarities, ranks are 1..k
        for r in recs:
            self.assertTrue(-1.0 <= r["score"] <= 1.0)
            self.assertIn(r["movie_id"], self.rec.movies.index)
        self.assertEqual([r["rank"] for r in recs], list(range(1, len(recs) + 1)))
        # recommendations must exclude items the user already interacted with
        u = self.rec.u_map[1]
        seen = set(self.rec.train[self.rec.train.u == u]["i"].tolist())
        seen_ids = {self.rec.idx2movie[i] for i in seen}
        self.assertTrue(all(r["movie_id"] not in seen_ids for r in recs))
        print(f"\n[serving] recommend(user=1, k={k}) -> {len(recs)} items "
              f"in {latency_ms:.1f} ms")

    def test_unknown_user_raises(self):
        with self.assertRaises(UnknownUserError):
            self.rec.recommend(user_id=10_000_000)

    def test_default_policy_never_reorders_retrieval(self):
        out = self.rec.recommend(user_id=1, k=10, n_candidates=50,
                                 return_candidates=True)
        recall_order = out["recall_order"]
        ranked_order = out["ranked_order"]
        self.assertEqual(recall_order, ranked_order)
        self.assertEqual(len(set(recall_order)), len(recall_order))
        self.assertLessEqual(len(recall_order), 50)
        self.assertEqual(ranked_order[:len(out["recommendations"])],
                         [r["movie_id"] for r in out["recommendations"]])

    def test_deepfm_ranking_remains_an_explicit_experiment(self):
        out = self.deepfm_rec.recommend(
            user_id=1, k=10, n_candidates=50, return_candidates=True)
        self.assertEqual(out["ranking_policy"], "deepfm")
        self.assertEqual(out["score_type"], "positive_rating_probability")
        self.assertIsNotNone(self.deepfm_rec.deepfm)
        self.assertEqual(sorted(out["recall_order"]),
                         sorted(out["ranked_order"]))
        self.assertEqual(out["ranked_order"][:len(out["recommendations"])],
                         [r["movie_id"] for r in out["recommendations"]])

    def test_heavy_user_does_not_overflow_the_faiss_index(self):
        """已看过的物品很多时，n_candidates+len(seen) 会超过库存量。

        Faiss 在 k > ntotal 时用 -1 填充；哨兵值必须被挡在候选之外，
        否则 idx2movie[-1] 会 KeyError。这里直接要一个大到必然越界的候选池。
        """
        out = self.rec.recommend(user_id=1, k=5,
                                 n_candidates=self.rec.n_items + 500,
                                 return_candidates=True)
        self.assertTrue(all(mid in self.rec.movies.index
                            for mid in out["recall_order"]))
        self.assertGreater(len(out["recommendations"]), 0)


class FieldOrderGuardTests(unittest.TestCase):
    """训练侧改了特征顺序，服务侧必须启动即失败而不是静默错分。"""

    def test_serving_pins_the_deepfm_field_order(self):
        import serving
        from train_deepfm import FIELD_COLS
        self.assertEqual(tuple(FIELD_COLS), serving.EXPECTED_FIELD_COLS)

    def test_history_cache_limit_is_positive(self):
        import serving
        self.assertGreater(serving.HISTORY_CACHE_LIMIT, 0)


if __name__ == "__main__":
    unittest.main()
