"""CPU-only tests for the minimal active recommendation tool loop."""
import unittest

import numpy as np
import torch

import train_dqn_rec as env_mod
from train_dqn_rec import RecSimEnv

from research_v3.agentic_rec.env import (
    AGENT_STATE_DIM,
    TOOL_ACTION,
    AgenticRecEnv,
)
from research_v3.agentic_rec.policy import (
    clipped_surrogate_loss,
    group_relative_advantages,
    kl_from_logits,
)
from research_v3.agentic_rec.policy import ToolPolicy


class VectorUserModel:
    def __init__(self, scores):
        self.scores = torch.tensor(scores, dtype=torch.float32)

    def encode(self, seq):
        return torch.zeros(1, seq.shape[1], 64)

    def score(self, ctx, item_ids):
        return self.scores[: len(item_ids)]


class FixedRandom:
    def __init__(self, value):
        self.value = float(value)

    def random(self):
        return self.value


def make_base_env():
    top_items = np.array([1, 2, 3, 4], dtype=np.int64)
    item_genre = {1: 0, 2: 1, 3: 2, 4: 3}
    train_seq = [list(range(1, 30))]
    model = VectorUserModel([2.0, 1.8, 0.0, -1.0])
    return RecSimEnv(model, 50, (top_items, item_genre, train_seq), seed=0)


class AgenticRecEnvTests(unittest.TestCase):
    def setUp(self):
        self.saved_device = env_mod.DEVICE
        env_mod.DEVICE = "cpu"

    def tearDown(self):
        env_mod.DEVICE = self.saved_device

    def test_reset_returns_augmented_state_and_no_initial_tool_gain(self):
        env = AgenticRecEnv(make_base_env())
        state = env.reset()
        self.assertEqual(state.shape, (AGENT_STATE_DIM,))
        self.assertEqual(state.dtype, np.float32)
        self.assertEqual(env.current_plan.baseline_action, 0)
        self.assertEqual(env.current_plan.tool_action, 0)
        self.assertAlmostEqual(env.current_plan.expected_gain, 0.0)

    def test_fatigue_aware_tool_switches_to_another_genre(self):
        env = AgenticRecEnv(make_base_env())
        env.reset()
        env.base.last_genre = 0
        env.base.run_len = 2
        _, scores = env.base._score(env.base.hist)
        plan = env._make_plan(scores)
        self.assertEqual(plan.baseline_action, 0)
        self.assertEqual(plan.tool_action, 1)
        self.assertGreater(plan.expected_gain, 0.01)

    def test_bonus_requires_useful_tool_and_successful_outcome(self):
        env = AgenticRecEnv(
            make_base_env(), tool_cost=0.03, tool_bonus=0.05, min_gain=0.01
        )
        base_state = env.reset()
        env.base.last_genre = 0
        env.base.run_len = 2
        ctx, scores = env.base._score(env.base.hist)
        env.current_plan = env._make_plan(scores)
        env._current_state = env._augment(env.base._state(ctx), env.current_plan)
        env.base.rng = FixedRandom(0.0)

        _, reward, _, info = env.step(TOOL_ACTION)
        self.assertTrue(info["tool_changed_item"])
        self.assertTrue(info["beneficial_tool"])
        self.assertTrue(info["bonus_earned"])
        self.assertAlmostEqual(info["user_reward"], 0.9)
        self.assertAlmostEqual(info["net_reward"], 0.87)
        self.assertAlmostEqual(reward, 0.92)
        self.assertEqual(base_state.shape, (AGENT_STATE_DIM,))

    def test_unnecessary_tool_pays_cost_without_bonus(self):
        env = AgenticRecEnv(
            make_base_env(), tool_cost=0.03, tool_bonus=0.05, min_gain=0.01
        )
        env.reset()
        env.base.rng = FixedRandom(0.0)
        _, reward, _, info = env.step(TOOL_ACTION)
        self.assertFalse(info["tool_changed_item"])
        self.assertFalse(info["beneficial_tool"])
        self.assertFalse(info["bonus_earned"])
        self.assertAlmostEqual(reward, 0.87)

    def test_invalid_meta_action_is_rejected(self):
        env = AgenticRecEnv(make_base_env())
        env.reset()
        with self.assertRaises(ValueError):
            env.step(2)


class GroupRelativeAdvantageTests(unittest.TestCase):
    def test_policy_accepts_full_agent_state_but_uses_decision_features(self):
        policy = ToolPolicy()
        full_state = torch.zeros(3, AGENT_STATE_DIM)
        logits = policy(full_state)
        self.assertEqual(tuple(logits.shape), (3, 2))

    def test_advantages_are_centered_and_scaled(self):
        advantages = group_relative_advantages([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(float(advantages.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(advantages.std()), 1.0, places=6)

    def test_identical_returns_have_zero_advantage(self):
        advantages = group_relative_advantages([2.0, 2.0, 2.0])
        np.testing.assert_array_equal(advantages, np.zeros(3, dtype=np.float32))

    def test_single_return_is_not_a_group(self):
        with self.assertRaises(ValueError):
            group_relative_advantages([1.0])


class GrpoLossTests(unittest.TestCase):
    def test_clip_binds_surrogate_for_large_ratio(self):
        # logp_new - logp_old ≈ 2.3 → ratio ≈ 10；正优势下裁剪到 1+ε=1.2
        logp_new = torch.tensor([0.0])
        logp_old = torch.tensor([-2.3])
        advantages = torch.tensor([1.0])
        loss = clipped_surrogate_loss(logp_new, logp_old, advantages, clip_eps=0.2)
        self.assertAlmostEqual(float(loss), -1.2, places=4)

    def test_clip_does_not_bind_inside_interval(self):
        logp_new = torch.tensor([0.0])
        logp_old = torch.tensor([0.05])
        advantages = torch.tensor([2.0])
        loss = clipped_surrogate_loss(logp_new, logp_old, advantages, clip_eps=0.2)
        ratio = float((logp_new - logp_old).exp())
        self.assertAlmostEqual(float(loss), -ratio * 2.0, places=5)

    def test_negative_advantage_penalty_is_not_capped(self):
        # 标准 PPO 语义：裁剪封顶"收益"但不封顶"惩罚"。优势为负且 ratio 很大时，
        # min(r·A, clip(r)·A) 取未裁剪项 r·A，loss = r·|A| ≈ 9.97。
        logp_new = torch.tensor([0.0])
        logp_old = torch.tensor([-2.3])
        advantages = torch.tensor([-1.0])
        loss = clipped_surrogate_loss(logp_new, logp_old, advantages, clip_eps=0.2)
        ratio = float((logp_new - logp_old).exp())
        self.assertAlmostEqual(float(loss), ratio, places=4)

    def test_kl_is_zero_for_identical_policies(self):
        logits = torch.tensor([[1.0, 0.5], [0.2, -1.0]])
        self.assertAlmostEqual(float(kl_from_logits(logits, logits)), 0.0, places=6)

    def test_kl_is_nonnegative(self):
        left = torch.tensor([[2.0, 0.1]])
        right = torch.tensor([[0.1, 2.0]])
        self.assertGreaterEqual(float(kl_from_logits(left, right)), 0.0)


if __name__ == "__main__":
    unittest.main()
