"""Policy and small mathematical helpers for active tool selection."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import AGENT_FEATURE_DIM, AGENT_STATE_DIM, N_META_ACTIONS


class ToolPolicy(nn.Module):
    """A deliberately small and interpretable baseline/tool gate.

    ``AgenticRecEnv`` keeps the full RecSim state for future extensions, but the
    minimum viable gate uses only the four explicit decision features appended
    to it: baseline probability, tool probability, expected gain, and fatigue
    run length.  This prevents the 84-dimensional user embedding from drowning
    out the cost/benefit signal in a tiny experiment.
    """

    def __init__(self, state_dim: int = AGENT_FEATURE_DIM) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.network = nn.Sequential(
            nn.Linear(self.state_dim, 16),
            nn.LayerNorm(16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, N_META_ACTIONS),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.shape[-1] == AGENT_STATE_DIM and self.state_dim == AGENT_FEATURE_DIM:
            states = states[..., -AGENT_FEATURE_DIM:]
        if states.shape[-1] != self.state_dim:
            raise ValueError(
                f"policy expected {self.state_dim} features, got {states.shape[-1]}"
            )
        return self.network(states)


def group_relative_advantages(returns, eps: float = 1e-8) -> np.ndarray:
    """Normalize rollout returns inside one same-context group.

    This is the group-relative part of the minimal experiment.  It is not full
    GRPO: there is no old-policy importance ratio or clipped surrogate here.
    """

    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("at least two one-dimensional returns are required")
    std = float(values.std())
    if std < eps:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - values.mean()) / (std + eps)).astype(np.float32)


def clipped_surrogate_loss(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float,
) -> torch.Tensor:
    """GRPO/PPO 的裁剪 surrogate：min(r·A, clip(r,1±ε)·A) 的负均值。

    r = exp(logp_new - logp_old) 是旧策略的重要性采样比。裁剪让每次更新
    "只改一点点"：即使旧数据里某个动作的回报很高，新策略也不能一步跳太远。
    """
    ratio = (logp_new - logp_old).exp()
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages
    return -torch.min(surr1, surr2).mean()


def kl_from_logits(
    logits: torch.Tensor,
    reference_logits: torch.Tensor,
) -> torch.Tensor:
    """逐状态 KL(π_new ‖ π_ref)，按动作求和后取平均。"""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    log_reference = F.log_softmax(reference_logits, dim=-1)
    return (policy * (log_policy - log_reference)).sum(dim=-1).mean()
