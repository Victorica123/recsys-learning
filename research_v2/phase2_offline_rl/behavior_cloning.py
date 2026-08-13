# -*- coding: utf-8 -*-
"""SFT 阶段：行为克隆（Behavior Cloning）。

类比 LLM 后训练：SFT = 监督式模仿"好"的示范。在推荐里，就是让策略网络预测
日志中"实际推了哪个物品"——**纯监督分类，不看奖励**。

两种变体：
- 普通 BC：交叉熵模仿全部日志动作，忠实复刻行为策略（连它的次优一起复刻）。
- 奖励加权 BC（RWR / filtered BC）：用整条会话的 return-to-go 给样本加权，
  只重点模仿"高长期回报"的示范——正是 LLM 里"拒绝采样 SFT / best-of-N"的思路，
  是从纯 SFT 迈向 RL 的第一步（策略提升，但仍不需要 Bellman 自举）。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCPolicy(nn.Module):
    """状态 → 动作 logits 的 MLP 策略网络。"""

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


def return_weights(returns: np.ndarray, temp: float) -> np.ndarray:
    """把 return-to-go 变成非负样本权重：exp((R - max)/temp)，均值归一。

    temp 越小越"挑食"（只学最高回报的示范），temp→∞ 退化成普通 BC（等权）。
    减去 max 仅为数值稳定，不改变相对权重。
    """
    r = np.asarray(returns, dtype=np.float64) / max(temp, 1e-6)
    r = r - r.max()
    w = np.exp(r)
    return (w / (w.mean() + 1e-8)).astype(np.float32)


def train_bc(dataset, n_actions: int, epochs: int = 5, lr: float = 1e-3,
             batch: int = 256, reward_weighted: bool = False,
             weight_temp: float = 5.0, device: str = "cpu", seed: int = 0):
    """在离线数据上做行为克隆。返回 (模型, 每轮平均损失列表)。"""
    rng = np.random.default_rng(seed)
    model = BCPolicy(dataset.state_dim, n_actions).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    weights = (return_weights(dataset.returns, weight_temp)
               if reward_weighted else None)
    n = len(dataset)
    steps_per_epoch = max(1, n // batch)
    history = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        for _ in range(steps_per_epoch):
            idx = rng.integers(0, n, size=batch)
            s = torch.as_tensor(dataset.states[idx], dtype=torch.float32,
                                device=device)
            a = torch.as_tensor(dataset.actions[idx], dtype=torch.int64,
                                device=device)
            logits = model(s)
            loss_i = F.cross_entropy(logits, a, reduction="none")
            if weights is not None:
                w = torch.as_tensor(weights[idx], dtype=torch.float32,
                                    device=device)
                loss = (loss_i * w).mean()
            else:
                loss = loss_i.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        history.append(total / steps_per_epoch)
    return model, history


@torch.no_grad()
def bc_accuracy(model, dataset, device: str = "cpu", n: int = 4096) -> float:
    """在数据集上抽样计算"预测动作 == 日志动作"的比例（模仿保真度）。"""
    m = min(n, len(dataset))
    s = torch.as_tensor(dataset.states[:m], dtype=torch.float32, device=device)
    pred = model(s).argmax(1).cpu().numpy()
    return float((pred == dataset.actions[:m]).mean())


def bc_action_fn(model, device: str = "cpu"):
    """把 BC 策略包装成 run_policy 需要的 policy_fn(env, state) -> action。"""
    def act(env, state):
        with torch.no_grad():
            t = torch.as_tensor(state, dtype=torch.float32, device=device)
            return int(model(t).argmax(1).item())
    return act
