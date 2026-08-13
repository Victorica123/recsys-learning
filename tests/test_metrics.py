# -*- coding: utf-8 -*-
"""Smoke tests for the hand-written ranking metric (`train_deepfm.auc_score`).

The DeepFM report headline (test AUC 0.754) rides entirely on this rank-statistic
implementation, so pin its contract against a brute-force pairwise reference.
"""
import unittest

import numpy as np

from train_deepfm import auc_score


def brute_force_auc(y_true, y_score):
    """Reference AUC: probability a random positive outranks a random negative."""
    pos = [i for i in range(len(y_true)) if y_true[i] == 1]
    neg = [i for i in range(len(y_true)) if y_true[i] == 0]
    wins = 0.0
    for p in pos:
        for n in neg:
            if y_score[p] > y_score[n]:
                wins += 1.0
            elif y_score[p] == y_score[n]:
                wins += 0.5
    return wins / (len(pos) * len(neg))


class AucScoreTests(unittest.TestCase):
    def test_perfect_ranking_is_one(self):
        y = np.array([1, 1, 0, 0])
        s = np.array([0.9, 0.8, 0.2, 0.1])
        self.assertAlmostEqual(auc_score(y, s), 1.0, places=6)

    def test_inverted_ranking_is_zero(self):
        y = np.array([1, 1, 0, 0])
        s = np.array([0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(auc_score(y, s), 0.0, places=6)

    def test_matches_brute_force_on_random_distinct_scores(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            n = int(rng.integers(4, 40))
            # Guarantee at least one positive and one negative label.
            y = rng.integers(0, 2, size=n)
            if y.min() == y.max():
                y[0] = 1 - y[0]
            # Distinct scores so rank-statistic and pairwise counts agree exactly.
            s = rng.permutation(n).astype(np.float64)
            self.assertAlmostEqual(
                auc_score(y, s), brute_force_auc(y.tolist(), s.tolist()),
                places=6,
            )

    def test_is_invariant_to_monotonic_score_scaling(self):
        y = np.array([1, 0, 1, 0, 1])
        s = np.array([3.0, 1.0, 2.5, 0.5, 4.0])
        base = auc_score(y, s)
        self.assertAlmostEqual(auc_score(y, 10.0 * s + 7.0), base, places=6)


if __name__ == "__main__":
    unittest.main()
