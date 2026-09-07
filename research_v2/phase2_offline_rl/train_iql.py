# -*- coding: utf-8 -*-
"""IQL 实验：复用已固化的离线数据集，对照 SFT / naive DQN / CQL。

文献驱动的方向调整（详见 notes/14 与 notes/12）：
- CQL 的保守在本仿真器反有害（α↑ → 贴回短视行为分布）；
- IQL（Kostrikov et al. 2021）不用 max 自举，用 expectile V 替代，
  从源头避开 OOD 高估 → 不需要保守项，理论上应同时避开 CQL 塌缩
  与 naive DQN 的潜在高估。

用法：
  python research_v2/phase2_offline_rl/train_iql.py \
      --dataset experiments/offline_rl_phase2_dataset.npz \
      --steps 15000 --eval_sessions 300
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import set_seed, ExperimentConfig                       # noqa: E402
from train_dqn_rec import (RecSimEnv, load_user_model, load_rec_assets,  # noqa: E402
                           greedy_action, run_policy, N_ACTIONS, GAMMA, DEVICE)

from research_v2.phase2_offline_rl.dataset import OfflineDataset     # noqa: E402
from research_v2.phase2_offline_rl.behavior_cloning import (          # noqa: E402
    BCPolicy, bc_action_fn, bc_accuracy)
from research_v2.phase2_offline_rl.cql import CQLAgent               # noqa: E402
from research_v2.phase2_offline_rl.iql import IQLAgent               # noqa: E402


def load_bc_checkpoint(prefix, name, state_dim):
    path = ROOT / "checkpoints" / f"{prefix}_{name}.pt"
    if not path.exists():
        return None
    ck = torch.load(path, map_location=DEVICE, weights_only=True)
    model = BCPolicy(state_dim, N_ACTIONS).to(DEVICE)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


def load_cql_checkpoint(prefix, name, state_dim, cql_alpha):
    path = ROOT / "checkpoints" / f"{prefix}_{name}.pt"
    if not path.exists():
        return None
    ck = torch.load(path, map_location=DEVICE, weights_only=True)
    agent = CQLAgent(state_dim, N_ACTIONS, gamma=GAMMA, lr=1e-4,
                     cql_alpha=cql_alpha, device=DEVICE)
    agent.online.load_state_dict(ck["model"])
    agent.online.eval()
    return agent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt", default="checkpoints/sasrec_main.pt")
    ap.add_argument("--dataset", default="experiments/offline_rl_phase2_dataset.npz")
    ap.add_argument("--baseline-prefix", default="offline_rl_phase2",
                    help="SFT/naive/CQL 检查点前缀，用于同数据集完整对比")
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval_sessions", type=int, default=300)
    ap.add_argument("--log_epsilon", type=float, default=0.2)
    ap.add_argument("--tag", default="iql-phase2")
    ap.add_argument("--no_eval", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="允许覆盖同名 checkpoint")
    a = ap.parse_args()
    set_seed(a.seed)

    cfg = ExperimentConfig.from_args(__file__, a, seed=a.seed)
    cfg.run_name = f"offline_rl_{a.tag}"
    cfg.save(ROOT / "experiments")
    print(f"实验配置已保存: {cfg.run_name}  (device={DEVICE})")

    dataset = OfflineDataset.load(ROOT / a.dataset)
    sdim = dataset.state_dim
    print(f"复用数据集 {a.dataset}  {dataset.reward_stats()}")

    # ---------- 训练 IQL ----------
    print(f"训练 IQL：τ={a.tau} β={a.beta} lr={a.lr}，{a.steps} steps ...",
          flush=True)
    agent = IQLAgent(sdim, N_ACTIONS, gamma=GAMMA, lr=a.lr, tau=a.tau,
                     beta=a.beta, device=DEVICE)
    rng = np.random.default_rng(a.seed)
    batch_size = min(256, len(dataset))
    t0 = time.time()
    log_path = ROOT / "experiments" / f"{cfg.run_name}_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["step", "q_loss", "v_loss",
                                          "policy_loss", "adv_mean",
                                          "q_mean", "elapsed_min"])
        w.writeheader()
        for step in range(1, a.steps + 1):
            stat = agent.learn(dataset.sample(batch_size, rng))
            if step % max(1, a.steps // 20) == 0 or step == a.steps:
                row = {"step": step,
                       "q_loss": round(stat["q_loss"], 4),
                       "v_loss": round(stat["v_loss"], 4),
                       "policy_loss": round(stat["policy_loss"], 4),
                       "adv_mean": round(stat["adv_mean"], 4),
                       "q_mean": round(stat["q_mean"], 3),
                       "elapsed_min": round((time.time() - t0) / 60, 2)}
                w.writerow(row)
                print(f"  step {step}/{a.steps}  Q={row['q_mean']:.2f} "
                      f"V损失={row['v_loss']:.3f} 策略损失={row['policy_loss']:.3f} "
                      f"优势均值={row['adv_mean']:.3f} | {row['elapsed_min']}min",
                      flush=True)

    ckpt_path = ROOT / "checkpoints" / f"{cfg.run_name}.pt"
    if ckpt_path.exists() and not a.force:
        raise FileExistsError(
            f"{ckpt_path.name} 已存在；用 --tag 新开实验或 --force 覆盖")
    torch.save(agent.state_dict(), ckpt_path)
    print(f"checkpoint -> {ckpt_path.name}")

    if a.no_eval:
        print("跳过评估（--no_eval）")
        return

    # ---------- 完整对比（同数据集历史基线 + IQL）----------
    sasrec, maxlen, _ = load_user_model(ROOT / a.ckpt)
    assets = load_rec_assets()
    mk_env = lambda: RecSimEnv(sasrec, maxlen, assets)
    eps = a.log_epsilon
    policies = {
        "随机策略": lambda e, s: e.rng.randrange(N_ACTIONS),
        "短视贪心": lambda e, s: greedy_action(e),
        "行为策略(ε-greedy短视)": (
            lambda e, s: (e.rng.randrange(N_ACTIONS)
                          if e.rng.random() < eps else greedy_action(e))),
    }
    bc = load_bc_checkpoint(a.baseline_prefix, "bc", sdim)
    bc_rwr = load_bc_checkpoint(a.baseline_prefix, "bc_rwr", sdim)
    naive = load_cql_checkpoint(a.baseline_prefix, "naive", sdim, 0.0)
    cql = load_cql_checkpoint(a.baseline_prefix, "cql", sdim, 1.0)
    if bc is not None:
        policies["SFT 行为克隆"] = bc_action_fn(bc, DEVICE)
    if bc_rwr is not None:
        policies["奖励加权 BC"] = bc_action_fn(bc_rwr, DEVICE)
    if naive is not None:
        policies["naive 离线 DQN"] = naive.act_fn()
    if cql is not None:
        policies["CQL 离线 RL"] = cql.act_fn()
    policies["IQL 离线 RL"] = agent.act_fn()

    rows = []
    for name, fn in policies.items():
        R, L, Dv = run_policy(mk_env(), fn, n_sessions=a.eval_sessions)
        rows.append((name, R, L, Dv))
        print(f"  {name:<22} 奖励 {R:6.2f} | 长度 {L:5.1f} | 多样性 {Dv:.2f}",
              flush=True)
    eval_path = ROOT / "experiments" / f"{cfg.run_name}_eval.csv"
    with open(eval_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["policy", "avg_reward", "avg_len", "genre_diversity"])
        for name, R, L, Dv in sorted(rows, key=lambda x: -x[1]):
            w.writerow([name, round(R, 2), round(L, 2), round(Dv, 2)])
    print(f"\n[{cfg.run_name}] 策略对比（{a.eval_sessions} 会话）已写入 "
          f"{eval_path.name}")
    print(f"BC 模仿保真度: {bc_accuracy(agent.policy, dataset, DEVICE):.2%}")


if __name__ == "__main__":
    main()
