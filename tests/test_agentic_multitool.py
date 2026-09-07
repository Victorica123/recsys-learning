"""CPU tests for budgeted multi-tool trajectories."""
import unittest

import numpy as np
import torch

import train_dqn_rec as env_mod
from train_dqn_rec import RecSimEnv

from research_v3.agentic_rec.multitool import (
    DIVERSITY_TOOL,
    FATIGUE_TOOL,
    MULTITOOL_STATE_DIM,
    PREFERENCE_TOOL,
    STOP_AND_SERVE,
    MultiToolPolicy,
    MultiToolRecEnv,
    masked_logits,
)


class VectorUserModel:
    def __init__(self, scores):
        self.scores = torch.tensor(scores, dtype=torch.float32)

    def encode(self, seq):
        return torch.zeros(1, seq.shape[1], 64)

    def score(self, ctx, item_ids):
        return self.scores[: len(item_ids)]


def make_base_env(seed=0):
    top_items = np.array([1, 2, 3, 4], dtype=np.int64)
    item_genre = {1: 0, 2: 1, 3: 2, 4: 3}
    train_seq = [list(range(1, 30))]
    model = VectorUserModel([2.0, 1.8, 0.0, -1.0])
    return RecSimEnv(model, 50, (top_items, item_genre, train_seq), seed=seed)


class MultiToolEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.saved_device = env_mod.DEVICE
        env_mod.DEVICE = "cpu"

    def tearDown(self):
        env_mod.DEVICE = self.saved_device

    def test_reset_exposes_all_affordable_actions(self):
        env = MultiToolRecEnv(make_base_env(), session_budget=0.60)
        state = env.reset()
        self.assertEqual(state.shape, (MULTITOOL_STATE_DIM,))
        self.assertEqual(env.action_mask().tolist(), [True, True, True, True])

    def test_tool_call_costs_budget_without_advancing_user_session(self):
        env = MultiToolRecEnv(make_base_env(), session_budget=0.20)
        env.reset()
        before = len(env.base.recent)
        _, reward, done, info = env.step(FATIGUE_TOOL)
        self.assertFalse(done)
        self.assertEqual(len(env.base.recent), before)
        self.assertAlmostEqual(reward, -0.05)
        self.assertAlmostEqual(env.remaining_budget, 0.15)
        self.assertTrue(info["tool_used"])

    def test_repeat_call_is_invalid_and_forces_stop_next(self):
        env = MultiToolRecEnv(make_base_env())
        env.reset()
        env.step(DIVERSITY_TOOL)
        _, reward, done, info = env.step(DIVERSITY_TOOL)
        self.assertFalse(done)
        self.assertTrue(info["invalid_action"])
        self.assertAlmostEqual(reward, -env.invalid_action_penalty)
        self.assertEqual(env.action_mask().tolist(), [True, False, False, False])

    def test_budget_masks_expensive_preference_tool(self):
        env = MultiToolRecEnv(make_base_env(), session_budget=0.08)
        env.reset()
        mask = env.action_mask()
        self.assertTrue(mask[FATIGUE_TOOL])
        self.assertTrue(mask[DIVERSITY_TOOL])
        self.assertFalse(mask[PREFERENCE_TOOL])

    def test_two_tools_then_stop_records_multi_tool_decision(self):
        env = MultiToolRecEnv(make_base_env(), max_tools_per_decision=2)
        env.reset()
        env.step(DIVERSITY_TOOL)
        env.step(PREFERENCE_TOOL)
        self.assertEqual(env.action_mask().tolist(), [True, False, False, False])
        _, _, _, info = env.step(STOP_AND_SERVE)
        self.assertTrue(info["decision_complete"])
        self.assertTrue(info["multi_tool_decision"])
        self.assertEqual(info["decision_tool_calls"], 2)


class MultiToolPolicyTests(unittest.TestCase):
    def test_policy_accepts_full_state(self):
        policy = MultiToolPolicy()
        logits = policy(torch.zeros(2, MULTITOOL_STATE_DIM))
        self.assertEqual(tuple(logits.shape), (2, 4))

    def test_masked_logits_never_select_invalid_action(self):
        logits = torch.tensor([[0.0, 100.0, 50.0, 10.0]])
        mask = torch.tensor([[True, False, False, False]])
        constrained = masked_logits(logits, mask)
        self.assertEqual(int(constrained.argmax(dim=1)), STOP_AND_SERVE)

    def test_empty_action_mask_is_rejected(self):
        with self.assertRaises(ValueError):
            masked_logits(torch.zeros(1, 4), torch.zeros(1, 4, dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()
