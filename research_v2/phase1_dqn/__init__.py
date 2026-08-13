"""Phase 1: Hand-written DQN from scratch.

This module builds on Phase 0 (tabular Q-learning on GridWorld) and replaces
the Q-table with a neural network.  The environment also graduates from the
4x4 grid to OpenAI Gymnasium's CartPole-v1.
"""

from .dqn_agent import DQNAgent, compute_td_target
from .q_network import QNetwork
from .replay_buffer import Batch, ReplayBuffer

__all__ = [
    "QNetwork",
    "ReplayBuffer",
    "Batch",
    "DQNAgent",
    "compute_td_target",
]
