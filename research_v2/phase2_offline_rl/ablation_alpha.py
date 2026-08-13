# -*- coding: utf-8 -*-
"""消融：CQL 保守系数 α 的权衡曲线（复用已固化的离线数据集，不重采）。

α 是离线 RL 最关键的旋钮：它在"信任数据里的行为策略"（α 大 → 贴近 BC）与
"信任 Bellman 做策略提升"（α=0 → 纯离线 DQN）之间插值。

本脚本在**同一份数据集**上扫 α，评估每个 α 训出的策略，量化：
  - α 太大 → 过度保守 → 策略被拉回（短视的）行为分布 → 奖励/多样性下降
  - α 太小/为 0 → 若数据覆盖足够好，纯 Bellman 反而能学到交错策略、拿高回报
  - 数据覆盖越差，越需要保守（见 notes/12 对"外推误差"的讨论）

用法：
  python research_v2/phase2_offline_rl/ablation_alpha.py \
      --dataset experiments/offline_rl_phase2_dataset.npz
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import set_seed                                          # noqa: E402
from train_dqn_rec import (RecSimEnv, load_user_model, load_rec_assets,  # noqa: E402
                           greedy_action, run_policy, N_ACTIONS, GAMMA, DEVICE)
from dataset import OfflineDataset                                   # noqa: E402
from cql import CQLAgent                                             # noqa: E402


def train_and_eval(dataset, alpha, rl_steps, lr, mk_env, eval_sessions, seed):
    rng = np.random.default_rng(seed)
    agent = CQLAgent(dataset.state_dim, N_ACTIONS, gamma=GAMMA, lr=lr,
                     cql_alpha=alpha, device=DEVICE)
    last_q = 0.0
    batch_size = min(256, len(dataset))
    for _ in range(rl_steps):
        last_q = agent.learn(dataset.sample(batch_size, rng))["q_mean"]
    R, L, Dv = run_policy(mk_env(), agent.act_fn(), n_sessions=eval_sessions)
    return R, L, Dv, last_q


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt", default="checkpoints/sasrec_main.pt")
    ap.add_argument("--dataset", default="experiments/offline_rl_phase2_dataset.npz")
    ap.add_argument("--alphas", default="0,0.5,1,2,5")
    ap.add_argument("--rl_steps", type=int, default=12000)
    ap.add_argument("--rl_lr", type=float, default=1e-4)
    ap.add_argument("--eval_sessions", type=int, default=200)
    a = ap.parse_args()
    set_seed(a.seed)

    dataset = OfflineDataset.load(ROOT / a.dataset)
    print(f"复用数据集 {a.dataset}  {dataset.reward_stats()}  (device={DEVICE})")
    sasrec, maxlen, _ = load_user_model(ROOT / a.ckpt)
    assets = load_rec_assets()
    mk_env = lambda: RecSimEnv(sasrec, maxlen, assets)

    # 参考基线（与 α 无关，各评一次）
    refs = [
        ("随机策略", run_policy(mk_env(), lambda e, s: e.rng.randrange(N_ACTIONS),
                            a.eval_sessions)),
        ("短视贪心", run_policy(mk_env(), lambda e, s: greedy_action(e),
                            a.eval_sessions)),
    ]

    rows = []
    for alpha in [float(x) for x in a.alphas.split(",")]:
        R, L, Dv, q = train_and_eval(dataset, alpha, a.rl_steps, a.rl_lr,
                                     mk_env, a.eval_sessions, a.seed)
        tag = "naive DQN" if alpha == 0 else f"CQL α={alpha:g}"
        rows.append((tag, alpha, R, L, Dv, q))
        print(f"  {tag:<12} 奖励 {R:6.2f} | 长度 {L:5.1f} | "
              f"多样性 {Dv:.2f} | 训练末 Q均值 {q:7.2f}", flush=True)

    out = ROOT / "experiments" / "offline_rl_alpha_sweep.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["policy", "cql_alpha", "avg_reward", "avg_len",
                    "genre_diversity", "final_q_mean"])
        for tag, alpha, R, L, Dv, q in rows:
            w.writerow([tag, alpha, round(R, 2), round(L, 2), round(Dv, 2),
                        round(q, 2)])
        for name, (R, L, Dv) in refs:
            w.writerow([name, "", round(R, 2), round(L, 2), round(Dv, 2), ""])
    print(f"\n参考: 随机 {refs[0][1][0]:.2f} | 短视贪心 {refs[1][1][0]:.2f}")
    print(f"结果已写入 {out.name}")


if __name__ == "__main__":
    main()
