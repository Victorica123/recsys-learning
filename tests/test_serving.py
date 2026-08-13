# -*- coding: utf-8 -*-
"""Tests for the reusable serving core (`src/serving.py`).

The pure helpers (`pad_genre_batch`, `check_artifacts`) run everywhere with no
model load. The end-to-end recommender test actually loads the two-tower and
DeepFM checkpoints, so it is opt-in via ``RECSYS_INTEGRATION=1`` to keep the
default smoke suite instant.
"""
import os
import time
import unittest

import torch

from serving import (Recommender, UnknownUserError, check_artifacts,
                     pad_genre_batch)


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


@unittest.skipUnless(os.environ.get("RECSYS_INTEGRATION") == "1",
                     "set RECSYS_INTEGRATION=1 to load real checkpoints")
class RecommenderIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rec = Recommender.load()

    def test_recommend_returns_k_valid_unseen_items(self):
        k = 10
        t0 = time.perf_counter()
        out = self.rec.recommend(user_id=1, k=k)
        latency_ms = (time.perf_counter() - t0) * 1000
        recs = out["recommendations"]
        self.assertLessEqual(len(recs), k)
        self.assertGreater(len(recs), 0)
        # scores are probabilities, ranks are 1..k, ids resolve to real movies
        for r in recs:
            self.assertTrue(0.0 <= r["score"] <= 1.0)
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


if __name__ == "__main__":
    unittest.main()
