"""PPO building blocks on synthetic trajectories (CPU, no environment)."""
from __future__ import annotations

import unittest

import numpy as np

from train_ppo_rec import compute_gae


class GaeTests(unittest.TestCase):
    def test_lambda_one_with_zero_value_matches_monte_carlo(self):
        rewards = np.array([1.0, -0.1, 0.9], dtype=np.float32)
        values = np.zeros(3, dtype=np.float32)
        dones = np.array([0, 0, 1], dtype=np.float32)
        adv, returns = compute_gae(rewards, values, dones, gamma=0.9, lam=1.0)
        expected = [1.639, 0.71, 0.9]  # 折扣回报 G_t
        for got, want in zip(returns.tolist(), expected):
            self.assertAlmostEqual(float(got), want, places=5)
        # λ=1 且 V≡0 时 GAE 优势等于回报本身
        np.testing.assert_allclose(adv, returns, atol=1e-6)

    def test_terminal_done_resets_bootstrap(self):
        rewards = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        dones = np.array([0, 1, 0], dtype=np.float32)
        adv, returns = compute_gae(rewards, values, dones, gamma=0.9, lam=0.5)
        # 手工推导：t=2 A=-2.7；t=1 终止步不再看 s2 的 V，A=-1.8；
        # t=0 A=0.9 + 0.9*0.5*(-1.8) = 0.09
        expected_adv = [0.09, -1.8, -2.7]
        for got, want in zip(adv.tolist(), expected_adv):
            self.assertAlmostEqual(float(got), want, places=5)
        expected_returns = [1.09, 0.2, 0.3]
        for got, want in zip(returns.tolist(), expected_returns):
            self.assertAlmostEqual(float(got), want, places=5)

    def test_lambda_zero_is_one_step_td(self):
        rewards = np.array([0.5, 1.0], dtype=np.float32)
        values = np.array([2.0, 3.0], dtype=np.float32)
        dones = np.array([0, 1], dtype=np.float32)
        adv, _ = compute_gae(rewards, values, dones, gamma=0.9, lam=0.0)
        # t=0: r0 + γ·V1 − V0 = 0.5 + 0.9*3 − 2 = 1.2
        self.assertAlmostEqual(float(adv[0]), 1.2, places=5)
        # t=1（终止）: r1 − V1 = 1 − 3 = −2
        self.assertAlmostEqual(float(adv[1]), -2.0, places=5)


if __name__ == "__main__":
    unittest.main()
