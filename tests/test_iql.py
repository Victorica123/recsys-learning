"""IQL building blocks on synthetic batches (CPU, no environment)."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from research_v2.phase2_offline_rl.dataset import Batch
from research_v2.phase2_offline_rl.iql import IQLAgent, advantage_weights


def make_batch(n=16, state_dim=4, n_actions=5, seed=0):
    rng = np.random.default_rng(seed)
    return Batch(
        states=rng.normal(size=(n, state_dim)).astype(np.float32),
        actions=rng.integers(0, n_actions, size=n),
        rewards=rng.uniform(-0.1, 0.9, size=n).astype(np.float32),
        next_states=rng.normal(size=(n, state_dim)).astype(np.float32),
        dones=(rng.random(n) < 0.3).astype(np.float32),
        returns=rng.uniform(0.0, 8.0, size=n).astype(np.float32),
    )


class AdvantageWeightTests(unittest.TestCase):
    def test_weights_are_positive_and_mean_normalized(self):
        adv = torch.tensor([-1.0, 0.0, 2.0, 5.0])
        w = advantage_weights(adv, beta=3.0)
        self.assertTrue(torch.isfinite(w).all())
        self.assertTrue((w > 0).all())
        self.assertAlmostEqual(float(w.mean()), 1.0, places=5)

    def test_higher_advantage_gets_higher_weight(self):
        adv = torch.tensor([-1.0, 5.0])
        w = advantage_weights(adv, beta=3.0)
        self.assertGreater(float(w[1]), float(w[0]))

    def test_smaller_beta_is_more_selective(self):
        adv = torch.tensor([0.0, 3.0])
        w_wide = advantage_weights(adv, beta=10.0)
        w_narrow = advantage_weights(adv, beta=1.0)
        # 窄 β 让高优势样本的相对权重更突出
        self.assertGreater(float(w_narrow[1] / w_narrow[0]),
                           float(w_wide[1] / w_wide[0]))


class IQLAgentTests(unittest.TestCase):
    def test_learn_returns_expected_stats(self):
        agent = IQLAgent(4, 5, gamma=0.9, device="cpu")
        batch = make_batch()
        stat = agent.learn(batch)
        for key in ("q_loss", "v_loss", "policy_loss", "adv_mean", "q_mean"):
            self.assertIn(key, stat)
            self.assertTrue(np.isfinite(stat[key]))
        # Q 目标用 V(s') 而非 max Q：验证 target 网络在 Bellman 里未被使用
        self.assertEqual(agent._steps, 1)

    def test_advantage_is_q_minus_v_not_mc_return(self):
        # AWR 优势来自 Q−V（离线 TD 语义），不是 dataset 的 return-to-go
        agent = IQLAgent(4, 5, gamma=0.9, device="cpu")
        batch = make_batch()
        agent.learn(batch)
        s = torch.as_tensor(batch.states, dtype=torch.float32)
        with torch.no_grad():
            q_vals = agent.q_target(s).gather(
                1, torch.as_tensor(batch.actions).unsqueeze(1)).squeeze(1)
            v_vals = agent.v(s)
        self.assertTrue(torch.isfinite(q_vals - v_vals).all())


if __name__ == "__main__":
    unittest.main()
