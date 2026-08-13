"""Tests for the DQN building blocks (Steps 2-4)."""

import unittest

import numpy as np
import torch

from .dqn_agent import DQNAgent, compute_td_target
from .replay_buffer import Batch, ReplayBuffer


class ReplayBufferTests(unittest.TestCase):
    def _fill(self, buffer: ReplayBuffer, n: int) -> None:
        for i in range(n):
            buffer.push(
                state=np.array([i, i, i, i], dtype=np.float32),
                action=i % 2,
                reward=1.0,
                next_state=np.array([i + 1] * 4, dtype=np.float32),
                done=False,
            )

    def test_capacity_evicts_oldest(self):
        buffer = ReplayBuffer(capacity=3)
        self._fill(buffer, 5)
        self.assertEqual(len(buffer), 3)

    def test_sample_shapes_and_types(self):
        buffer = ReplayBuffer(capacity=100)
        self._fill(buffer, 50)
        rng = np.random.default_rng(0)
        batch = buffer.sample(16, rng)
        self.assertIsInstance(batch, Batch)
        self.assertEqual(batch.states.shape, (16, 4))
        self.assertEqual(batch.actions.shape, (16,))
        self.assertEqual(batch.next_states.shape, (16, 4))
        self.assertEqual(batch.actions.dtype, np.int64)
        self.assertEqual(batch.rewards.dtype, np.float32)

    def test_sample_too_many_raises(self):
        buffer = ReplayBuffer(capacity=10)
        self._fill(buffer, 4)
        with self.assertRaises(ValueError):
            buffer.sample(8, np.random.default_rng(0))

    def test_sampling_is_deterministic_given_rng(self):
        buffer = ReplayBuffer(capacity=100)
        self._fill(buffer, 50)
        a = buffer.sample(8, np.random.default_rng(7))
        b = buffer.sample(8, np.random.default_rng(7))
        np.testing.assert_array_equal(a.states, b.states)


class TDTargetTests(unittest.TestCase):
    def test_terminal_transition_ignores_bootstrap(self):
        # done=1 -> target must equal the immediate reward only.
        target = compute_td_target(
            rewards=torch.tensor([2.0]),
            next_max_q=torch.tensor([100.0]),
            dones=torch.tensor([1.0]),
            gamma=0.99,
        )
        self.assertAlmostEqual(target.item(), 2.0, places=5)

    def test_non_terminal_bootstraps(self):
        target = compute_td_target(
            rewards=torch.tensor([1.0]),
            next_max_q=torch.tensor([10.0]),
            dones=torch.tensor([0.0]),
            gamma=0.9,
        )
        self.assertAlmostEqual(target.item(), 1.0 + 0.9 * 10.0, places=5)


class DQNAgentTests(unittest.TestCase):
    def test_select_action_in_range(self):
        agent = DQNAgent(state_dim=4, n_actions=2)
        rng = np.random.default_rng(0)
        state = np.zeros(4, dtype=np.float32)
        for epsilon in (0.0, 1.0):
            action = agent.select_action(state, epsilon, rng)
            self.assertIn(action, (0, 1))

    def test_learn_returns_finite_loss_and_updates_weights(self):
        agent = DQNAgent(state_dim=4, n_actions=2, lr=1e-2)
        before = agent.online.net[0].weight.detach().clone()
        rng = np.random.default_rng(0)
        buffer = ReplayBuffer(capacity=200)
        for _ in range(64):
            buffer.push(
                state=rng.standard_normal(4).astype(np.float32),
                action=int(rng.integers(2)),
                reward=1.0,
                next_state=rng.standard_normal(4).astype(np.float32),
                done=bool(rng.integers(2)),
            )
        loss = agent.learn(buffer.sample(32, rng))
        self.assertTrue(np.isfinite(loss))
        after = agent.online.net[0].weight.detach()
        self.assertFalse(torch.equal(before, after), "weights did not update")

    def test_hard_sync_copies_online_into_target(self):
        agent = DQNAgent(state_dim=4, n_actions=2, target_sync=1, lr=1e-1)
        rng = np.random.default_rng(1)
        buffer = ReplayBuffer(capacity=100)
        for _ in range(40):
            buffer.push(
                np.zeros(4, np.float32), 0, 1.0, np.ones(4, np.float32), False
            )
        agent.learn(buffer.sample(32, rng))  # target_sync=1 -> sync after this call
        for p_online, p_target in zip(
            agent.online.parameters(), agent.target.parameters()
        ):
            self.assertTrue(torch.equal(p_online, p_target))


if __name__ == "__main__":
    unittest.main()
