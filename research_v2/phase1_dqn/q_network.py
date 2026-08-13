# -*- coding: utf-8 -*-
"""Step 1: Q-Network -- a tiny MLP that replaces the Q-table.

Why a neural network?
  Phase 0 had 16 states x 4 actions = 64 entries -- easy to store in a table.
  CartPole's state is 4 continuous numbers -- infinite possibilities.
  We need a function approximator: input state, output Q-value for each action.

Architecture:
  state (4 floats) -> Linear(4,64) -> ReLU -> Linear(64,64) -> ReLU -> Linear(64,2)
                                                                           ^^^^^
                                                                  Q(s, left), Q(s, right)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """A small feed-forward network that maps state -> Q-values.

    Parameters
    ----------
    state_dim : int
        Number of dimensions in the observation (CartPole = 4).
    n_actions : int
        Number of discrete actions (CartPole = 2: left / right).
    hidden_dim : int
        Width of the hidden layers (default 64).
    """

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return Q-values for every action.

        Parameters
        ----------
        state : torch.Tensor
            Shape (batch_size, state_dim) or (state_dim,).

        Returns
        -------
        torch.Tensor
            Shape (batch_size, n_actions) -- one Q-value per action.
        """
        # Make sure we always have a batch dimension.
        if state.dim() == 1:
            state = state.unsqueeze(0)  # (4,) -> (1, 4)
        return self.net(state)


# ---------------------------------------------------------------------------
# Sanity checks (run with pytest or directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Quick manual test: feed a random CartPole observation.
    net = QNetwork(state_dim=4, n_actions=2)
    dummy_state = torch.randn(4)  # single state
    q_values = net(dummy_state)
    print(f"Input  shape: {dummy_state.shape}")
    print(f"Output shape: {q_values.shape}")  # expected: (1, 2)
    print(f"Q-values: {q_values}")

    # Batch of 3 states
    batch = torch.randn(3, 4)
    q_batch = net(batch)
    print(f"\nBatch input  shape: {batch.shape}")
    print(f"Batch output shape: {q_batch.shape}")  # expected: (3, 2)
    print(f"Batch Q-values:\n{q_batch}")
