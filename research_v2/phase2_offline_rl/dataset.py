# -*- coding: utf-8 -*-
"""离线数据集：把"生产日志"固化成固定数据，训练时完全不碰环境。

真实推荐系统无法在线自由探索——你只有一批历史曝光/反馈日志。离线 RL 的全部
难点都源于此：策略要从**固定分布**的数据里学习，且不能对数据里没出现过的动作
盲目乐观（否则 Bellman 自举会把高估无限放大）。

本模块与具体环境解耦：`collect_dataset` 接收任意 duck-typed 环境
（需实现 `reset()` 与 `step(a) -> (s, r, done, info)`）和一个行为策略函数，
滚动记录五元组 `(s, a, r, s', done)`，并回填每步的**折扣 return-to-go**
（整条会话的长期回报，供奖励加权 SFT 使用）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from typing import NamedTuple


class Batch(NamedTuple):
    """一个 mini-batch 的转移，每个字段形状 (batch_size, ...)。"""

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    returns: np.ndarray   # 折扣 return-to-go（该步之后整条会话的长期回报）


class OfflineDataset:
    """固定的离线转移数据集，支持随机采样、存盘/读盘。"""

    def __init__(self, states, actions, rewards, next_states, dones, returns):
        self.states = np.asarray(states, dtype=np.float32)
        self.actions = np.asarray(actions, dtype=np.int64)
        self.rewards = np.asarray(rewards, dtype=np.float32)
        self.next_states = np.asarray(next_states, dtype=np.float32)
        self.dones = np.asarray(dones, dtype=np.float32)
        self.returns = np.asarray(returns, dtype=np.float32)
        n = len(self.states)
        for name, arr in [("actions", self.actions), ("rewards", self.rewards),
                          ("next_states", self.next_states), ("dones", self.dones),
                          ("returns", self.returns)]:
            if len(arr) != n:
                raise ValueError(f"{name} 长度 {len(arr)} 与 states {n} 不一致")

    def __len__(self):
        return len(self.states)

    @property
    def state_dim(self) -> int:
        return int(self.states.shape[1])

    def sample(self, batch_size: int, rng: np.random.Generator) -> Batch:
        """均匀有放回采样一个 mini-batch。"""
        if len(self) < batch_size:
            raise ValueError(
                f"cannot sample {batch_size} from dataset of size {len(self)}")
        idx = rng.integers(0, len(self), size=batch_size)
        return Batch(self.states[idx], self.actions[idx], self.rewards[idx],
                     self.next_states[idx], self.dones[idx], self.returns[idx])

    def reward_stats(self) -> dict:
        """数据集画像：规模、即时奖励均值、长期回报均值、动作覆盖度。"""
        return {
            "n_transitions": len(self),
            "reward_mean": float(self.rewards.mean()),
            "return_mean": float(self.returns.mean()),
            "return_max": float(self.returns.max()),
            "unique_actions": int(len(np.unique(self.actions))),
        }

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, states=self.states, actions=self.actions,
            rewards=self.rewards, next_states=self.next_states,
            dones=self.dones, returns=self.returns)
        return path

    @classmethod
    def load(cls, path) -> "OfflineDataset":
        d = np.load(path)
        return cls(d["states"], d["actions"], d["rewards"],
                   d["next_states"], d["dones"], d["returns"])


def collect_dataset(env, policy_fn, n_sessions: int,
                    gamma: float = 0.9, seed: int = 0) -> OfflineDataset:
    """用 `policy_fn(env, state, rng)` 在 env 里滚动 n_sessions 条会话记录日志。

    policy_fn 是"行为策略"（生产环境里正在跑的那个策略），离线数据就是它的日志。
    返回带 return-to-go 的 OfflineDataset。
    """
    rng = np.random.default_rng(seed)
    states, actions, rewards, next_states, dones, returns = [], [], [], [], [], []
    for _ in range(n_sessions):
        s = env.reset()
        traj = []
        while True:
            a = policy_fn(env, s, rng)
            s2, r, done, _ = env.step(a)
            traj.append((np.asarray(s, dtype=np.float32), int(a), float(r),
                         np.asarray(s2, dtype=np.float32), bool(done)))
            s = s2
            if done:
                break
        # 从会话末尾往前累计折扣 return-to-go
        g = 0.0
        rtg = [0.0] * len(traj)
        for t in range(len(traj) - 1, -1, -1):
            g = traj[t][2] + gamma * g
            rtg[t] = g
        for (s_, a_, r_, s2_, d_), gt in zip(traj, rtg):
            states.append(s_); actions.append(a_); rewards.append(r_)
            next_states.append(s2_); dones.append(d_); returns.append(gt)
    return OfflineDataset(states, actions, rewards, next_states, dones, returns)
