"""Step 3: DQN Agent -- the same Bellman update as Phase 0, now with two nets.

Recall the tabular update from Phase 0::

    target = r + gamma * max_a' Q(s', a')      # 0 if s' is terminal
    Q(s, a) <- Q(s, a) + alpha * (target - Q(s, a))

DQN keeps the *target* identical but changes the mechanics:

  1. ``Q(s, a)`` is now a neural network, not a table lookup.
  2. ``max_a' Q(s', a')`` is computed by a **separate, frozen target network**.
     If we bootstrapped off the same weights we are updating, the target would
     chase the prediction and training would diverge (the "deadly triad" that
     the recsys DQN track diagnosed at length).
  3. The "alpha step" becomes a gradient-descent step on the squared TD error,
     averaged over a random mini-batch from the replay buffer.

The target network is refreshed by a **hard copy** every ``target_sync`` steps --
the simplest, most stable choice for CartPole.

Overestimation and Double DQN
-----------------------------
Plain DQN evaluates the next state with ``max_a' Q_target(s', a')``. Because the
same network both *picks* and *scores* the best action, any positive noise in a
Q-estimate is systematically selected and propagated -- Q-values inflate and, on
CartPole, the greedy policy peaks and then collapses once exploration stops.

Double DQN (van Hasselt et al., 2016) breaks this coupling: the **online** net
chooses the next action, the **target** net scores it. That single change is
enough to keep CartPole stable, so it is the default here (``double_dqn=True``).
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .q_network import QNetwork


def compute_td_target(
    rewards: torch.Tensor,
    next_max_q: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Vectorised Bellman target: ``r + gamma * max_a' Q(s',a') * (1 - done)``.

    Kept as a free function so it can be unit-tested without a network, exactly
    like ``q_learning_update`` in Phase 0.
    """
    return rewards + gamma * next_max_q * (1.0 - dones)


class DQNAgent:
    """Deep Q-Network agent for a discrete-action environment.

    Parameters
    ----------
    state_dim, n_actions : int
        Environment dimensions (CartPole = 4, 2).
    gamma : float
        Discount factor.
    lr : float
        Adam learning rate.
    target_sync : int
        Hard-copy the online weights into the target net every this many
        ``learn`` calls.
    double_dqn : bool
        Use Double DQN targets (online net selects, target net evaluates).
    device : str
        "cpu" or "cuda".
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        gamma: float = 0.99,
        lr: float = 1e-3,
        target_sync: int = 500,
        double_dqn: bool = True,
        device: str = "cpu",
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.target_sync = target_sync
        self.double_dqn = double_dqn
        self.device = torch.device(device)

        self.online = QNetwork(state_dim, n_actions).to(self.device)
        self.target = copy.deepcopy(self.online).to(self.device)
        self.target.eval()  # target net is never trained directly
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self._learn_steps = 0

    @torch.no_grad()
    def select_action(
        self, state: np.ndarray, epsilon: float, rng: np.random.Generator
    ) -> int:
        """Epsilon-greedy action selection (identical policy shape to Phase 0)."""
        if rng.random() < epsilon:
            return int(rng.integers(self.n_actions))
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        q_values = self.online(state_t)  # (1, n_actions)
        return int(q_values.argmax(dim=1).item())

    def learn(self, batch) -> float:
        """One gradient step on a mini-batch; returns the scalar loss.

        ``batch`` is a :class:`~replay_buffer.Batch` of NumPy arrays.
        """
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(
            batch.next_states, dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device)

        # Q(s, a) for the actions we actually took.
        q_taken = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # max_a' Q_target(s', a') -- no gradient through the target net.
        with torch.no_grad():
            next_max_q = self.target(next_states).max(dim=1).values
            td_target = compute_td_target(rewards, next_max_q, dones, self.gamma)

        # Smooth L1 (Huber) is the robust default DQN loss.
        loss = F.smooth_l1_loss(q_taken, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()

        self._learn_steps += 1
        if self._learn_steps % self.target_sync == 0:
            self.sync_target()
        return float(loss.item())

    def sync_target(self) -> None:
        """Hard-copy online weights into the target network."""
        self.target.load_state_dict(self.online.state_dict())
