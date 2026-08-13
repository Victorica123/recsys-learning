# -*- coding: utf-8 -*-
"""Smoke tests for the offline-RL post-training stack (research_v2/phase2_offline_rl).

All tests run on tiny synthetic data — no SASRec checkpoint, no simulator, no GPU.
They pin the three load-bearing invariants:
  1. Dataset logging computes discounted return-to-go correctly.
  2. Behavior cloning (SFT) actually imitates the logged actions.
  3. CQL's conservative term pushes down out-of-distribution action values —
     the whole reason offline RL needs conservatism.
"""
import unittest

import numpy as np
import torch

from research_v2.phase2_offline_rl.dataset import (OfflineDataset,
                                                   collect_dataset)
from research_v2.phase2_offline_rl.behavior_cloning import (train_bc,
                                                            bc_accuracy,
                                                            return_weights)
from research_v2.phase2_offline_rl.cql import CQLAgent


class _StubEnv:
    """Fixed-length episode env: reward 1 each step, done after ep_len steps."""

    def __init__(self, state_dim=4, ep_len=3):
        self.state_dim = state_dim
        self.ep_len = ep_len
        self.t = 0

    def reset(self):
        self.t = 0
        return np.zeros(self.state_dim, dtype=np.float32)

    def step(self, a):
        self.t += 1
        s2 = np.full(self.state_dim, self.t, dtype=np.float32)
        return s2, 1.0, self.t >= self.ep_len, {}


class DatasetTests(unittest.TestCase):
    def test_return_to_go_is_discounted_correctly(self):
        ds = collect_dataset(_StubEnv(ep_len=3), lambda e, s, rng: 0,
                             n_sessions=1, gamma=0.9, seed=0)
        self.assertEqual(len(ds), 3)
        # rewards [1,1,1], gamma 0.9 -> rtg = [2.71, 1.9, 1.0]
        np.testing.assert_allclose(ds.returns, [2.71, 1.9, 1.0], atol=1e-5)

    def test_shapes_and_sampling(self):
        ds = collect_dataset(_StubEnv(state_dim=5, ep_len=4),
                             lambda e, s, rng: 1, n_sessions=3, seed=1)
        self.assertEqual(ds.state_dim, 5)
        self.assertEqual(len(ds), 12)
        batch = ds.sample(6, np.random.default_rng(0))
        self.assertEqual(batch.states.shape, (6, 5))
        self.assertEqual(batch.returns.shape, (6,))

    def test_save_load_roundtrip(self):
        import tempfile
        import os
        ds = collect_dataset(_StubEnv(), lambda e, s, rng: 0, n_sessions=2)
        with tempfile.TemporaryDirectory() as d:
            path = ds.save(os.path.join(d, "ds.npz"))
            back = OfflineDataset.load(path)
        np.testing.assert_array_equal(ds.states, back.states)
        np.testing.assert_array_equal(ds.returns, back.returns)


class ReturnWeightTests(unittest.TestCase):
    def test_weights_are_nonnegative_and_mean_normalized(self):
        w = return_weights(np.array([0.0, 1.0, 2.0, 3.0]), temp=1.0)
        self.assertTrue((w >= 0).all())
        self.assertAlmostEqual(float(w.mean()), 1.0, places=4)

    def test_higher_return_gets_higher_weight(self):
        w = return_weights(np.array([0.0, 5.0]), temp=1.0)
        self.assertGreater(w[1], w[0])


class BehaviorCloningTests(unittest.TestCase):
    def _onehot_dataset(self, n_actions=4, n=400, seed=0):
        """State = one-hot(action) + noise, so action is fully recoverable."""
        rng = np.random.default_rng(seed)
        acts = rng.integers(0, n_actions, size=n)
        states = np.eye(n_actions, dtype=np.float32)[acts]
        states += 0.05 * rng.standard_normal((n, n_actions)).astype(np.float32)
        return OfflineDataset(states, acts, np.ones(n), states,
                              np.zeros(n), np.ones(n))

    def test_bc_learns_to_imitate_logged_actions(self):
        ds = self._onehot_dataset()
        model, hist = train_bc(ds, n_actions=4, epochs=15, lr=5e-3, batch=64,
                               device="cpu", seed=0)
        self.assertLess(hist[-1], hist[0])                 # loss went down
        self.assertGreater(bc_accuracy(model, ds), 0.9)    # imitation fidelity


class CQLConservatismTests(unittest.TestCase):
    def _single_action_dataset(self, state_dim=6, n=512, seed=0):
        """Only action 0 ever appears -> actions 1..A are out-of-distribution."""
        rng = np.random.default_rng(seed)
        states = rng.standard_normal((n, state_dim)).astype(np.float32)
        return OfflineDataset(states, np.zeros(n, dtype=np.int64),
                              np.ones(n), states, np.zeros(n), np.ones(n))

    def test_cql_suppresses_out_of_distribution_action_values(self):
        ds = self._single_action_dataset()
        rng = np.random.default_rng(0)
        agent = CQLAgent(ds.state_dim, n_actions=5, cql_alpha=5.0,
                         lr=1e-3, device="cpu")
        for _ in range(300):
            agent.learn(ds.sample(128, rng))
        q = agent.q_values(ds.states[:256])           # (256, 5)
        q_data_action = q[:, 0].mean()                # the only logged action
        q_ood_actions = q[:, 1:].mean()               # never-seen actions
        # Conservatism: unseen actions must not be valued above the seen one.
        self.assertGreater(q_data_action, q_ood_actions)

    def test_learn_reports_expected_stats(self):
        ds = self._single_action_dataset(n=128)
        agent = CQLAgent(ds.state_dim, n_actions=5, cql_alpha=1.0, device="cpu")
        stat = agent.learn(ds.sample(64, np.random.default_rng(0)))
        for key in ("loss", "td_loss", "cql_term", "q_mean"):
            self.assertIn(key, stat)
            self.assertTrue(np.isfinite(stat[key]))


if __name__ == "__main__":
    unittest.main()
