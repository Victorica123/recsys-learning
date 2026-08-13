# -*- coding: utf-8 -*-
"""离线 RL：隐式 Q 学习（IQL, Kostrikov et al. 2021, arXiv:2110.06169）。

为什么在 CQL 之后引入 IQL（文献驱动的方向调整）：

1. CQL 在本仿真器的教训（notes/12）：回报有界 + 所有动作合法时，"压制 OOD
   高估"的保守项没有用武之地，反而把策略拉回短视的行为分布（α↑ → 贴 BC）。
2. IQL 走完全不同的路线：**根本不对未见动作做 max**。Bellman 目标里用
   expectile 回归出的 V(s') 代替 `max_a' Q(s',a')`，从源头避开"max 挑中
   网络幻觉动作"的外推误差，因此不需要保守项。
3. 策略用**优势加权行为克隆（AWR）**提取：只放大"Q 明显高于 V"的数据动作，
   天然贴合数据分布，不依赖数据里没见过的动作。

三件套（离散动作版）：
    Q(s,a)  ← TD:  r + γ(1-done)·V(s')            （V 替换 max，无 OOD max）
    V(s)    ← expectile 回归到 Q(s,a)（τ=0.7：偏乐观的分位数，而非均值）
    π(a|s)  ← exp((Q(s,a) − V(s))/β) 加权的交叉熵（AWR）
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cql import QNet
from .behavior_cloning import BCPolicy


class ValueNet(nn.Module):
    """状态 → 标量 V(s) 的 MLP（LayerNorm 稳定价值函数）。"""

    def __init__(self, state_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.net(state).squeeze(-1)


def advantage_weights(advantages: torch.Tensor, beta: float) -> torch.Tensor:
    """AWR 权重：exp(adv/β)，裁剪防溢出后均值归一。β→0 越"挑食"。"""
    w = (advantages / max(beta, 1e-6)).clamp(max=10.0).exp()
    return w / (w.mean() + 1e-8)


class IQLAgent:
    """IQL：Q + expectile V + 优势加权 BC，纯离线训练（只吃 batch，不碰环境）。"""

    def __init__(self, state_dim: int, n_actions: int, gamma: float = 0.9,
                 lr: float = 1e-4, tau: float = 0.7, beta: float = 3.0,
                 target_sync: int = 500, device: str = "cpu"):
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.beta = beta
        self.target_sync = target_sync
        self.device = torch.device(device)

        self.q = QNet(state_dim, n_actions).to(self.device)
        self.q_target = copy.deepcopy(self.q).to(self.device)
        self.q_target.eval()
        self.v = ValueNet(state_dim).to(self.device)
        self.policy = BCPolicy(state_dim, n_actions).to(self.device)

        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.v_opt = torch.optim.Adam(self.v.parameters(), lr=lr)
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self._steps = 0

    def learn(self, batch) -> dict:
        """一步梯度更新（Q / V / 策略 各自优化器），返回各项统计。"""
        d = self.device
        s = torch.as_tensor(batch.states, dtype=torch.float32, device=d)
        a = torch.as_tensor(batch.actions, dtype=torch.int64, device=d)
        r = torch.as_tensor(batch.rewards, dtype=torch.float32, device=d)
        s2 = torch.as_tensor(batch.next_states, dtype=torch.float32, device=d)
        done = torch.as_tensor(batch.dones, dtype=torch.float32, device=d)

        with torch.no_grad():
            # IQL 关键：Bellman 目标用 V(s')，不做 max_a' Q(s',a')
            next_v = self.v(s2)
            td_target = r + self.gamma * (1.0 - done) * next_v
            # V 的回归目标与策略优势都用"目标 Q"（稳定、去自举噪声）
            q_target_vals = self.q_target(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # ---- Q：TD 更新（smooth L1）----
        q_taken = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        q_loss = F.smooth_l1_loss(q_taken, td_target)
        self.q_opt.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.q_opt.step()

        # ---- V：expectile 回归到 Q_target(s,a) ----
        v_pred = self.v(s)
        diff = q_target_vals - v_pred
        weight = torch.where(diff > 0, self.tau, 1.0 - self.tau)
        v_loss = (weight * diff.square()).mean()
        self.v_opt.zero_grad()
        v_loss.backward()
        nn.utils.clip_grad_norm_(self.v.parameters(), 10.0)
        self.v_opt.step()

        # ---- 策略：优势加权行为克隆（AWR）----
        with torch.no_grad():
            advantages = q_target_vals - v_pred
            w = advantage_weights(advantages, self.beta)
        logits = self.policy(s)
        policy_loss = (F.cross_entropy(logits, a, reduction="none") * w).mean()
        self.policy_opt.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
        self.policy_opt.step()

        self._steps += 1
        if self._steps % self.target_sync == 0:
            self.q_target.load_state_dict(self.q.state_dict())  # 硬目标同步
        return {
            "q_loss": float(q_loss.item()),
            "v_loss": float(v_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "adv_mean": float(advantages.mean().item()),
            "q_mean": float(q_taken.mean().item()),
        }

    def act_fn(self):
        """包装成 run_policy 需要的 policy_fn(env, state) -> action（策略贪心）。"""
        def act(env, state):
            with torch.no_grad():
                t = torch.as_tensor(state, dtype=torch.float32,
                                    device=self.device)
                return int(self.policy(t).argmax(1).item())
        return act

    def state_dict(self):
        return {
            "q": self.q.state_dict(),
            "v": self.v.state_dict(),
            "policy": self.policy.state_dict(),
            "q_opt": self.q_opt.state_dict(),
            "v_opt": self.v_opt.state_dict(),
            "policy_opt": self.policy_opt.state_dict(),
            "tau": self.tau,
            "beta": self.beta,
        }
