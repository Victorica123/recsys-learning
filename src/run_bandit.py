# -*- coding: utf-8 -*-
"""
阶段 4：Contextual Bandit —— 推荐系统的"探索-利用"
====================================================

【业务问题】
一个短视频 App 推什么你点什么，越推越窄（信息茧房）——因为模型只敢推
"确定你喜欢"的（利用 exploitation），不敢试"你可能喜欢但没试过的"
（探索 exploration）。这就是推荐里的 RL 问题。

Bandit（老虎机）是 RL 的单步简化版：每轮选一个"臂"（这里是 18 种电影
类型），立刻得到奖励（用户喜不喜欢），没有长期状态转移。
- 随机策略：基线
- ε-greedy：90% 推最优类型，10% 随机探索（不看用户特征）
- LinUCB：为每个类型维护一个线性回归（输入用户特征），
  用"预测值 + 不确定度奖励"打分 —— 不确定的臂加分，
  这就是"乐观面对未知"（optimism in the face of uncertainty）

【环境怎么来？】用 MovieLens 真实数据统计出每个用户对每种类型的
喜欢概率 p(u,g)，模拟时按 p 掷硬币给奖励 —— 离线仿真是工业界
评估探索策略的标准第一步。

【运行方式】.venv/Scripts/python.exe src/run_bandit.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from config import set_seed, ExperimentConfig
from constants import GENRES

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "ml-1m"
EXP_DIR = ROOT / "experiments"
EXP_DIR.mkdir(exist_ok=True)

N_ARMS = len(GENRES)
T = 200000        # 模拟 20 万次推荐
ALPHA_SMOOTH = 4.0  # 喜欢概率的拉普拉斯平滑


# ---------------------------------------------------------------- 环境
def build_env(seed=42):
    """从真实数据构建：用户特征矩阵 X、喜欢概率表 P[u, g]。"""
    ratings = pd.read_csv(DATA / "ratings.dat", sep="::", engine="python",
                          names=["user_id", "movie_id", "rating", "timestamp"])
    users = pd.read_csv(DATA / "users.dat", sep="::", engine="python",
                        names=["user_id", "gender", "age", "occupation", "zip"])
    movies = pd.read_csv(DATA / "movies.dat", sep="::", engine="python",
                         names=["movie_id", "title", "genres"],
                         encoding="latin-1")

    ratings["like"] = (ratings["rating"] >= 4).astype(float)
    g2i = {g: i for i, g in enumerate(GENRES)}
    u_map = {u: i for i, u in enumerate(sorted(ratings["user_id"].unique()))}
    n_users = len(u_map)

    # 每个用户对每种类型的 喜欢次数/总次数 → 平滑后的喜欢概率
    like_cnt = np.zeros((n_users, N_ARMS))
    tot_cnt = np.zeros((n_users, N_ARMS))
    m2g = movies.set_index("movie_id")["genres"].to_dict()
    for r in ratings.itertuples():
        u = u_map[r.user_id]
        for g in m2g[r.movie_id].split("|"):
            tot_cnt[u, g2i[g]] += 1
            like_cnt[u, g2i[g]] += r.like
    user_mean = like_cnt.sum(1) / tot_cnt.sum(1).clip(1)
    P = (like_cnt + ALPHA_SMOOTH * user_mean[:, None]) \
        / (tot_cnt + ALPHA_SMOOTH)

    # 用户上下文特征：[bias, 性别2, 年龄7, 职业21] + [18维历史口味]
    # 口味特征只用每个用户【时间最早 30%】的评分统计（模拟"新用户冷启动
    # 一段时间后"的画像），避免直接泄露答案
    age_vals = sorted(users["age"].unique())
    occ_vals = sorted(users["occupation"].unique())
    d = 1 + 2 + len(age_vals) + len(occ_vals) + N_ARMS
    X = np.zeros((n_users, d))
    for r in users.itertuples():
        u = u_map[r.user_id]
        off = 1
        X[u, 0] = 1.0
        X[u, off + (r.gender == "F")] = 1.0
        off += 2
        X[u, off + age_vals.index(r.age)] = 1.0
        off += len(age_vals)
        X[u, off + occ_vals.index(r.occupation)] = 1.0

    rs = ratings.sort_values("timestamp")
    cnt = rs.groupby("user_id")["user_id"].transform("size")
    nth = rs.groupby("user_id").cumcount()
    early = rs[nth < (cnt * 0.3).clip(lower=1)]
    ecnt = np.zeros((n_users, N_ARMS))
    elike = np.zeros((n_users, N_ARMS))
    for r in early.itertuples():
        u = u_map[r.user_id]
        for g in m2g[r.movie_id].split("|"):
            ecnt[u, g2i[g]] += 1
            elike[u, g2i[g]] += r.like
    taste = (elike + 2.0 * user_mean[:, None]) / (ecnt + 2.0)  # 平滑
    X[:, -N_ARMS:] = taste
    return X, P, u_map


# ---------------------------------------------------------------- 策略
class RandomPolicy:
    def choose(self, x): return np.random.randint(N_ARMS)
    def update(self, x, a, r): pass


class EpsilonGreedy:
    """只看历史平均奖励，10% 概率随机探索 —— 不认识"人"。"""
    def __init__(self, eps=0.1):
        self.eps = eps
        self.cnt = np.zeros(N_ARMS)
        self.val = np.zeros(N_ARMS)

    def choose(self, x):
        if np.random.rand() < self.eps or self.cnt.sum() == 0:
            return np.random.randint(N_ARMS)
        return int(np.argmax(self.val))

    def update(self, x, a, r):
        self.cnt[a] += 1
        self.val[a] += (r - self.val[a]) / self.cnt[a]  # 增量式均值


class LinUCB:
    """每个臂一个线性模型：打分 = 预测奖励 + alpha × 不确定度。

    工程细节：每一步都对 18 个臂求逆矩阵太慢，这里用 Sherman-Morrison
    公式直接维护逆矩阵 A_inv（秩一更新），复杂度从 O(d³) 降到 O(d²)，
    这也是 LinUCB 原始论文里的标准实现。
    """
    def __init__(self, d, alpha=1.0):
        self.alpha = alpha
        self.A_inv = np.stack([np.eye(d) for _ in range(N_ARMS)])
        self.b = np.zeros((N_ARMS, d))

    def choose(self, x):
        # theta = A_inv @ b；不确定度 = sqrt(xᵀ A_inv x)
        theta = np.einsum("aij,aj->ai", self.A_inv, self.b)
        mean = theta @ x
        unc = np.sqrt(np.einsum("i,aij,j->a", x, self.A_inv, x))
        return int(np.argmax(mean + self.alpha * unc))

    def update(self, x, a, r):
        Ai = self.A_inv[a]
        Ax = Ai @ x
        self.A_inv[a] -= np.outer(Ax, Ax) / (1.0 + x @ Ax)
        self.b[a] += r * x


# ---------------------------------------------------------------- 仿真
def simulate(policy, X, P, user_seq, rng):
    rewards = np.empty(len(user_seq))
    for t in range(len(user_seq)):
        u = user_seq[t]
        x = X[u]
        a = policy.choose(x)
        r = float(rng.random() < P[u, a])   # 按真实统计概率掷硬币
        policy.update(x, a, r)
        rewards[t] = r
    return rewards


class Frozen:
    """冻结策略的"纯利用"评估：不再探索、不再学习，只看策略本身质量。"""
    def __init__(self, policy):
        import copy
        self.pol = copy.deepcopy(policy)
        if hasattr(self.pol, "alpha"):
            self.pol.alpha = 0.0
        if hasattr(self.pol, "eps"):
            self.pol.eps = 0.0

    def choose(self, x):
        return self.pol.choose(x)

    def update(self, x, a, r):
        pass


def main():
    ap = argparse.ArgumentParser(description="Contextual Bandit ??")
    ap.add_argument("--seed", type=int, default=42, help="????")
    ap.add_argument("--tag", type=str, default="", help="????")
    args = ap.parse_args()
    set_seed(args.seed)
    cfg = ExperimentConfig.from_args(__file__, args, seed=args.seed)
    cfg.save(EXP_DIR)
    print(f"???????: {cfg.run_name}_config.json")

    X, P, _ = build_env(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    user_seq = rng.integers(0, len(X), T)   # 三个策略面对完全相同的用户流

    d = X.shape[1]
    policies = {"随机": RandomPolicy(),
                "ε-greedy": EpsilonGreedy(0.1),
                "LinUCB": LinUCB(d, alpha=0.5)}

    print(f"模拟 {T} 轮推荐（{len(X)} 个真实用户画像，{N_ARMS} 个类型臂）")
    print(f"参考：全局最优单一类型 CTR 上限 {P.mean(0).max():.3f} | "
          f"个性化 Oracle {P.max(1).mean():.3f}")
    print("=" * 62)

    eval_seq = np.random.default_rng(1).integers(0, len(X), 50000)
    log_path = EXP_DIR / "bandit_log.csv"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("policy,round,cum_reward,running_ctr\n")
        for name, pol in policies.items():
            np.random.seed(42)                # 保证公平：同样的探索随机性
            rewards = simulate(pol, X, P, user_seq,
                               np.random.default_rng(7))
            cum = np.cumsum(rewards)
            ctr = cum / (np.arange(T) + 1)
            # 冻结策略，在 5 万个新回合里纯利用，评估"学到的东西"本身
            frozen_r = simulate(Frozen(pol), X, P, eval_seq,
                                np.random.default_rng(2))
            print(f"{name:10s} | 在线学习 CTR {ctr[-1]:.3f} "
                  f"| 冻结后策略质量 {frozen_r.mean():.3f}")
            for t in range(0, T, 1000):
                f.write(f"{name},{t + 1},{cum[t]},{ctr[t]:.5f}\n")
    print("\n日志已写入:", log_path)


if __name__ == "__main__":
    main()
