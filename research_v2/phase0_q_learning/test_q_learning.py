"""Standard-library tests for the Phase 0 environment and algorithm."""

import unittest

import numpy as np

from .gridworld import RIGHT, UP, GridWorld
from .q_learning import evaluate_policy, q_learning_update, train_q_learning


class GridWorldTests(unittest.TestCase):
    def test_boundary_keeps_agent_in_grid(self):
        env = GridWorld()
        next_state, reward, done = env.transition(0, UP)
        self.assertEqual(next_state, 0)
        self.assertAlmostEqual(reward, -0.02)
        self.assertFalse(done)

    def test_goal_is_terminal(self):
        env = GridWorld()
        state_left_of_goal = env.position_to_state((3, 2))
        next_state, reward, done = env.transition(state_left_of_goal, RIGHT)
        self.assertEqual(next_state, env.goal_state)
        self.assertEqual(reward, 1.0)
        self.assertTrue(done)


class QLearningTests(unittest.TestCase):
    def test_one_bellman_update(self):
        q_table = np.zeros((3, 2), dtype=np.float64)
        q_table[1] = [0.4, 0.2]
        error = q_learning_update(
            q_table,
            state=0,
            action=1,
            reward=-0.02,
            next_state=1,
            done=False,
            alpha=0.1,
            gamma=0.95,
        )
        self.assertAlmostEqual(error, 0.36)
        self.assertAlmostEqual(q_table[0, 1], 0.036)

    def test_training_learns_a_reliable_policy(self):
        q_table, _ = train_q_learning(episodes=1_000, seed=7, eval_every=0)
        metrics = evaluate_policy(q_table, episodes=200, seed=8)
        self.assertGreaterEqual(metrics["success_rate"], 0.95)
        self.assertLessEqual(metrics["avg_steps"], 8.0)


if __name__ == "__main__":
    unittest.main()
