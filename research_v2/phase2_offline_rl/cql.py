# -*- coding: utf-8 -*-
"""离线 RL：保守 Q 学习（CQL, Kumar et al. 2020）。

离线 RL 的致命问题是**外推误差（extrapolation error）**：Bellman 目标里的
`max_a' Q(s',a')` 会挑中数据集里**没出现过的动作**，而这些 OOD 动作的 Q 值纯属
网络瞎猜、且被 max 系统性地选成"看起来最高"的那个；自举把高估一轮轮放大 →
Q 爆炸、策略崩坏。（在线 DQN 靠真实环境交互能纠偏，离线**没有这个机会**。）

CQL 的解法：给 TD 损失加一个"保守"正则项（离散动作版 CQL(H)）：

    L = L_TD + alpha * E_s[ logsumexp_a Q(s,a) - Q(s, a_data) ]
                        └────────── 压低所有动作 ──────┘ └ 抬高数据内动作 ┘

第一项压低"所有动作"的 Q（尤其被 max 挑中的 OOD 动作），第二项抬高"数据里真实
出现过的动作"的 Q → 学到的 Q 对没见过的动作保守，不再盲目乐观。
`cql_alpha=0` 时退化为 naive 离线 DQN（用于对照演示发散）。
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNet(nn.Module):
    """状态 → 每个动作 Q 值的 MLP（LayerNorm 稳定价值函数）。"""

    def __init__(self, state_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.net(state)


class CQLAgent:
    """Double DQN + CQL 保守正则，纯离线训练（只吃 batch，不碰环境）。

    Parameters
    ----------
    cql_alpha : float
        保守项权重。0 → naive 离线 DQN；越大越保守（越贴近行为策略）。
    """

    def __init__(self, state_dim: int, n_actions: int, gamma: float = 0.9,
                 lr: float = 1e-4, cql_alpha: float = 1.0,
                 target_sync: int = 500, device: str = "cpu"):
        self.n_actions = n_actions
        self.gamma = gamma
        self.cql_alpha = cql_alpha
        self.target_sync = target_sync
        self.device = torch.device(device)
        self.online = QNet(state_dim, n_actions).to(self.device)
        self.target = copy.deepcopy(self.online).to(self.device)
        self.target.eval()
        self.opt = torch.optim.Adam(self.online.parameters(), lr=lr)
        self._steps = 0

    def learn(self, batch) -> dict:
        """一步梯度更新，返回各项损失/统计（供画收敛曲线）。"""
        d = self.device
        s = torch.as_tensor(batch.states, dtype=torch.float32, device=d)
        a = torch.as_tensor(batch.actions, dtype=torch.int64, device=d)
        r = torch.as_tensor(batch.rewards, dtype=torch.float32, device=d)
        s2 = torch.as_tensor(batch.next_states, dtype=torch.float32, device=d)
        done = torch.as_tensor(batch.dones, dtype=torch.float32, device=d)

        q_all = self.online(s)                                   # (B, A)
        q_taken = q_all.gather(1, a.unsqueeze(1)).squeeze(1)     # (B,)

        with torch.no_grad():
            a_star = self.online(s2).argmax(1, keepdim=True)     # Double DQN：在线网选动作
            next_q = self.target(s2).gather(1, a_star).squeeze(1)  # 目标网估值
            td_target = r + self.gamma * (1.0 - done) * next_q
        td_loss = F.smooth_l1_loss(q_taken, td_target)

        # CQL(H) 保守项：logsumexp_a Q(s,a) - Q(s, a_data)
        logsumexp_q = torch.logsumexp(q_all, dim=1)              # (B,)
        cql_term = (logsumexp_q - q_taken).mean()
        loss = td_loss + self.cql_alpha * cql_term

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()

        self._steps += 1
        if self._steps % self.target_sync == 0:
            self.target.load_state_dict(self.online.state_dict())  # 硬目标同步
        return {"loss": float(loss.item()), "td_loss": float(td_loss.item()),
                "cql_term": float(cql_term.item()),
                "q_mean": float(q_all.mean().item())}

    @torch.no_grad()
    def q_values(self, states) -> np.ndarray:
        t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        return self.online(t).cpu().numpy()

    def act_fn(self):
        """包装成 run_policy 需要的 policy_fn(env, state) -> action（贪心）。"""
        def act(env, state):
            with torch.no_grad():
                t = torch.as_tensor(state, dtype=torch.float32,
                                    device=self.device)
                return int(self.online(t).argmax(1).item())
        return act

    def state_dict(self):
        return {"model": self.online.state_dict(),
                "opt": self.opt.state_dict(), "cql_alpha": self.cql_alpha}
