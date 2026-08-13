# -*- coding: utf-8 -*-
"""Tests for the Q-Network (Step 1)."""

import unittest

import torch

from .q_network import QNetwork


class QNetworkTests(unittest.TestCase):
    def test_output_shape_single_state(self):
        net = QNetwork(state_dim=4, n_actions=2)
        state = torch.randn(4)
        q = net(state)
        self.assertEqual(q.shape, (1, 2))

    def test_output_shape_batch(self):
        net = QNetwork(state_dim=4, n_actions=2)
        batch = torch.randn(8, 4)
        q = net(batch)
        self.assertEqual(q.shape, (8, 2))

    def test_different_hidden_dim(self):
        net = QNetwork(state_dim=4, n_actions=3, hidden_dim=32)
        state = torch.randn(4)
        q = net(state)
        self.assertEqual(q.shape, (1, 3))

    def test_gradient_flows(self):
        """Verify we can compute gradients through the network.
        If this fails, the network breaks autograd."""
        net = QNetwork(state_dim=4, n_actions=2)
        state = torch.randn(1, 4, requires_grad=True)
        q = net(state)
        loss = q.sum()
        loss.backward()
        # Check that at least one parameter got a gradient.
        grads = [p.grad is not None and p.grad.abs().sum() > 0
                 for p in net.parameters()]
        self.assertTrue(any(grads), "No parameter received a gradient")


if __name__ == "__main__":
    unittest.main()
