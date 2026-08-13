"""Step 2: Replay Buffer -- break the correlation between consecutive samples.

Why do we need it?
  In Phase 0 we updated the Q-table from each transition the moment it happened.
  A neural network trained that way sees a stream of *highly correlated* states
  (the cart drifts slowly, so s_t and s_{t+1} look almost identical). Gradient
  descent hates correlated mini-batches -- it forgets old experience and chases
  the latest trajectory. DQN's fix (Mnih et al., 2015) is to store transitions
  in a fixed-size buffer and train on *random* mini-batches drawn from it.

One transition is the same 5-tuple we used in Phase 0:
  (state, action, reward, next_state, done)

The buffer is deliberately framework-agnostic: it stores plain floats/ints and
hands back NumPy arrays. The agent is responsible for moving them onto a device.
"""

from __future__ import annotations

from collections import deque
from typing import NamedTuple

import numpy as np


class Batch(NamedTuple):
    """A mini-batch of transitions, each field shaped (batch_size, ...)."""

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray


class ReplayBuffer:
    """Fixed-capacity ring buffer of transitions with uniform random sampling.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions kept. Once full, the oldest is dropped.
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = capacity
        self._buffer: deque[tuple] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Append one transition; silently evicts the oldest when full."""
        self._buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
            )
        )

    def sample(self, batch_size: int, rng: np.random.Generator) -> Batch:
        """Draw ``batch_size`` transitions uniformly at random (with replacement).

        Sampling with replacement keeps the code simple and is standard for DQN
        once the buffer is large; the caller should only sample after the buffer
        holds at least ``batch_size`` transitions.
        """
        if len(self._buffer) < batch_size:
            raise ValueError(
                f"cannot sample {batch_size} from buffer of size {len(self._buffer)}"
            )
        indices = rng.integers(0, len(self._buffer), size=batch_size)
        states, actions, rewards, next_states, dones = zip(
            *(self._buffer[i] for i in indices)
        )
        return Batch(
            states=np.stack(states),
            actions=np.asarray(actions, dtype=np.int64),
            rewards=np.asarray(rewards, dtype=np.float32),
            next_states=np.stack(next_states),
            dones=np.asarray(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buffer)
