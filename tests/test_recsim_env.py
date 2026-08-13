# -*- coding: utf-8 -*-
"""Smoke tests for the recommendation-session MDP (`train_dqn_rec.RecSimEnv`).

The environment is the heart of the long-term-RL research track: fatigue on
repeated genres (``0.75 ** (run_len - 1)``) and churn after three consecutive
"no-feel" recommendations. Here we drive it with a *stub* user model instead of
the frozen SASRec simulator, so the transition dynamics are exercised on CPU
with no checkpoint, no dataset, and no GPU.
"""
import unittest

import numpy as np
import torch

import train_dqn_rec as env_mod
from train_dqn_rec import GENRES, N_ACTIONS, RecSimEnv, SESSION, STATE_DIM


class StubUserModel:
    """Stand-in for the frozen SASRec: returns a constant score per action."""

    def __init__(self, action_score, ctx_dim=64):
        self.action_score = float(action_score)
        self.ctx_dim = ctx_dim

    def encode(self, seq):
        length = seq.shape[1]
        return torch.zeros(1, length, self.ctx_dim)

    def score(self, ctx, item_ids):
        return torch.full((len(item_ids),), self.action_score)


class FakeRandom:
    """Deterministic replacement for the like/dislike coin flip."""

    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


def make_env(action_score):
    top_items = np.arange(1, N_ACTIONS + 1, dtype=np.int64)
    item_genre = {int(it): idx % len(GENRES) for idx, it in enumerate(top_items)}
    train_seq = [list(range(1, 30))]  # one long-enough sequence for reset()
    assets = (top_items, item_genre, train_seq)
    return RecSimEnv(StubUserModel(action_score), maxlen=50, assets=assets, seed=0)


class RecSimEnvTests(unittest.TestCase):
    def setUp(self):
        # Keep every tensor op on CPU regardless of CUDA availability.
        self._saved_device = env_mod.DEVICE
        env_mod.DEVICE = "cpu"

    def tearDown(self):
        env_mod.DEVICE = self._saved_device

    def test_reset_returns_state_of_expected_shape(self):
        env = make_env(action_score=0.0)
        state = env.reset()
        self.assertEqual(state.shape, (STATE_DIM,))
        self.assertEqual(state.dtype, np.float32)

    def test_like_gives_positive_reward_and_extends_history(self):
        env = make_env(action_score=50.0)  # score -> p ~= 1.0
        env.reset()
        env.rng = FakeRandom(0.0)           # coin flip always "like"
        before = len(env.hist)
        state, reward, done, info = env.step(0)
        self.assertTrue(info["like"])
        self.assertAlmostEqual(reward, 0.9)
        self.assertFalse(done)
        self.assertEqual(len(env.hist), before + 1)
        self.assertEqual(env.consec_dislike, 0)
        self.assertEqual(state.shape, (STATE_DIM,))

    def test_three_consecutive_dislikes_trigger_churn_exit(self):
        env = make_env(action_score=-50.0)  # score -> p ~= 0
        env.reset()
        env.rng = FakeRandom(1.0)           # coin flip always "dislike"
        dones = []
        for step in range(3):
            _, reward, done, info = env.step(step)  # distinct actions/genres
            self.assertFalse(info["like"])
            self.assertAlmostEqual(reward, -0.1)
            dones.append(done)
        self.assertEqual(dones, [False, False, True])
        self.assertLess(3, SESSION)  # churn beat the session-length cap

    def test_repeated_genre_decays_like_probability_by_fatigue_factor(self):
        env = make_env(action_score=0.0)    # base p = sigmoid(0) = 0.5
        env.reset()
        env.rng = FakeRandom(1.0)           # force dislikes so history stays fixed
        probs = [env.step(0)[3]["p"] for _ in range(3)]  # same action -> same genre
        self.assertAlmostEqual(probs[0], 0.5, places=6)
        self.assertAlmostEqual(probs[1], 0.75 * probs[0], places=6)
        self.assertAlmostEqual(probs[2], 0.75 * probs[1], places=6)


if __name__ == "__main__":
    unittest.main()
