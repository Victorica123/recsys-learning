# -*- coding: utf-8 -*-
"""推荐系统"后训练"流水线：采日志 → SFT（行为克隆）→ 离线 RL（CQL）→ 评估。

对齐大厂 LLM 后训练范式（SFT → RLHF），落在推荐的真实业务约束上：线上不能自由
探索，只能用固定日志离线训练。本脚本：
  1. 用 ε-greedy 短视策略在 RecSimEnv 采集"生产日志"（固定数据集）
  2. SFT：普通行为克隆 + 奖励加权行为克隆
  3. 离线 RL：CQL 保守 Q 学习，并对照 naive 离线 DQN（演示外推高估发散）
  4. 在 RecSimEnv 里评估各策略（仿真器充当现实中拿不到的"在线 A/B 预言机"）

用法：
  冒烟：  python research_v2/phase2_offline_rl/train.py --smoke
  完整：  python research_v2/phase2_offline_rl/train.py --sessions 2000 --rl_steps 20000
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import set_seed, ExperimentConfig                       # noqa: E402
from train_dqn_rec import (RecSimEnv, load_user_model, load_rec_assets,  # noqa: E402
                           greedy_action, run_policy, N_ACTIONS, GAMMA, DEVICE)

from dataset import collect_dataset, OfflineDataset                 # noqa: E402
from behavior_cloning import (train_bc, bc_action_fn, bc_accuracy)  # noqa: E402
from cql import CQLAgent                                            # noqa: E402


def behavior_policy_fn(epsilon, allowed=None):
    """行为策略（生产日志的来源）：ε 概率随机探索，否则短视贪心（即时最优）。

    这是一个"还不错但短视"的策略——恰好留给离线 RL 提升空间：短视贪心会连续推
    同类型直到疲劳，而最优策略需要交错类型。

    allowed 非空时，行为策略只在这批动作里选（其余动作成为数据里从未出现的 OOD
    动作）——用于制造"窄覆盖"日志，考验离线 RL 的外推鲁棒性。
    """
    if allowed is not None:
        allowed = np.array(sorted(allowed), dtype=np.int64)

        def act(env, state, rng):
            if rng.random() < epsilon:
                return int(rng.choice(allowed))
            _, scores = env._score(env.hist)          # 仅在允许集合内取即时最优
            return int(allowed[int(np.argmax(scores[allowed]))])
        return act

    def act(env, state, rng):
        if rng.random() < epsilon:
            return int(rng.integers(N_ACTIONS))
        return greedy_action(env)
    return act


def eval_policy_fns(epsilon):
    """评估用的三个基线 policy_fn(env, state)（run_policy 只传 env, state）。"""
    return {
        "随机策略": lambda e, s: e.rng.randrange(N_ACTIONS),
        "行为策略(ε-greedy短视)": (lambda e, s: (e.rng.randrange(N_ACTIONS)
                                    if e.rng.random() < epsilon
                                    else greedy_action(e))),
        "短视贪心": lambda e, s: greedy_action(e),
    }


def train_offline_agents(dataset, rl_steps, cql_alpha, lr, log_path, seed):
    """并行训练 CQL 与 naive 离线 DQN（同数据、同 batch 序列），记录 Q 均值发散对比。"""
    rng = np.random.default_rng(seed)
    sdim = dataset.state_dim
    cql = CQLAgent(sdim, N_ACTIONS, gamma=GAMMA, lr=lr,
                   cql_alpha=cql_alpha, device=DEVICE)
    naive = CQLAgent(sdim, N_ACTIONS, gamma=GAMMA, lr=lr,
                     cql_alpha=0.0, device=DEVICE)
    batch_size = min(256, len(dataset))
    rows = []
    for step in range(1, rl_steps + 1):
        # 关键：两个 agent 吃**完全相同**的 batch，唯一变量是 cql_alpha
        b = dataset.sample(batch_size, rng)
        c_stat = cql.learn(b)
        n_stat = naive.learn(b)
        if step % max(1, rl_steps // 20) == 0 or step == rl_steps:
            rows.append({"step": step,
                         "cql_q_mean": round(c_stat["q_mean"], 3),
                         "naive_q_mean": round(n_stat["q_mean"], 3),
                         "cql_td": round(c_stat["td_loss"], 4),
                         "cql_term": round(c_stat["cql_term"], 4)})
            print(f"  step {step}/{rl_steps}  CQL Q均值={c_stat['q_mean']:.2f}  "
                  f"naive Q均值={n_stat['q_mean']:.2f}", flush=True)
    if log_path:
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return cql, naive, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt", default="checkpoints/sasrec_main.pt")
    ap.add_argument("--sessions", type=int, default=2000, help="采集的会话数（日志规模）")
    ap.add_argument("--log_epsilon", type=float, default=0.2, help="行为策略探索率")
    ap.add_argument("--restrict_actions", type=int, default=0,
                    help=">0 时行为策略只用前 N 个动作，制造窄覆盖(其余为 OOD)")
    ap.add_argument("--bc_epochs", type=int, default=8)
    ap.add_argument("--rl_steps", type=int, default=20000)
    ap.add_argument("--cql_alpha", type=float, default=1.0)
    ap.add_argument("--rl_lr", type=float, default=1e-4)
    ap.add_argument("--eval_sessions", type=int, default=300)
    ap.add_argument("--tag", default="phase2")
    ap.add_argument("--no_eval", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="极小规模跑通全流程")
    a = ap.parse_args()
    if a.smoke:
        a.sessions, a.bc_epochs, a.rl_steps, a.eval_sessions = 60, 2, 300, 40
    set_seed(a.seed)

    cfg = ExperimentConfig.from_args(__file__, a, seed=a.seed)
    cfg.run_name = f"offline_rl_{a.tag}"
    cfg.save(ROOT / "experiments")
    print(f"实验配置已保存: {cfg.run_name}  (device={DEVICE})")

    # ---------- 1) 采集"生产日志" ----------
    print(f"[1/4] 加载 SASRec 用户模型并采集 {a.sessions} 条会话日志 ...", flush=True)
    sasrec, maxlen, _ = load_user_model(ROOT / a.ckpt)
    assets = load_rec_assets()
    mk_env = lambda: RecSimEnv(sasrec, maxlen, assets)
    t0 = time.time()
    allowed = list(range(a.restrict_actions)) if a.restrict_actions > 0 else None
    dataset = collect_dataset(mk_env(), behavior_policy_fn(a.log_epsilon, allowed),
                              n_sessions=a.sessions, gamma=GAMMA, seed=a.seed)
    ds_path = dataset.save(ROOT / "experiments" / f"{cfg.run_name}_dataset.npz")
    stats = dataset.reward_stats()
    print(f"      日志固化 -> {ds_path.name}  {stats}  "
          f"({time.time()-t0:.1f}s)", flush=True)

    # ---------- 2) SFT：行为克隆 ----------
    print(f"[2/4] SFT 行为克隆（普通 + 奖励加权），{a.bc_epochs} epochs ...", flush=True)
    bc, bc_hist = train_bc(dataset, N_ACTIONS, epochs=a.bc_epochs,
                           device=DEVICE, seed=a.seed)
    bc_rwr, rwr_hist = train_bc(dataset, N_ACTIONS, epochs=a.bc_epochs,
                                reward_weighted=True, device=DEVICE, seed=a.seed)
    print(f"      BC 损失 {bc_hist[0]:.3f}->{bc_hist[-1]:.3f}  "
          f"模仿保真度={bc_accuracy(bc, dataset, DEVICE):.2%}  "
          f"| RWR 损失 {rwr_hist[0]:.3f}->{rwr_hist[-1]:.3f}", flush=True)
    torch.save({"model": bc.state_dict()}, ROOT / "checkpoints" / f"{cfg.run_name}_bc.pt")
    torch.save({"model": bc_rwr.state_dict()}, ROOT / "checkpoints" / f"{cfg.run_name}_bc_rwr.pt")

    # ---------- 3) 离线 RL：CQL vs naive ----------
    print(f"[3/4] 离线 RL：CQL(alpha={a.cql_alpha}) vs naive DQN，{a.rl_steps} steps ...",
          flush=True)
    log_path = ROOT / "experiments" / f"{cfg.run_name}_log.csv"
    cql, naive, _ = train_offline_agents(dataset, a.rl_steps, a.cql_alpha,
                                         a.rl_lr, log_path, a.seed)
    torch.save(cql.state_dict(), ROOT / "checkpoints" / f"{cfg.run_name}_cql.pt")
    torch.save(naive.state_dict(), ROOT / "checkpoints" / f"{cfg.run_name}_naive.pt")

    if a.no_eval:
        print("跳过评估（--no_eval）"); return

    # ---------- 4) 仿真器评估（离线策略评估的"预言机"）----------
    print(f"[4/4] 在 RecSimEnv 评估各策略（{a.eval_sessions} 会话，同种子同会话流）...",
          flush=True)
    policies = dict(eval_policy_fns(a.log_epsilon))
    policies["SFT 行为克隆"] = bc_action_fn(bc, DEVICE)
    policies["奖励加权 BC"] = bc_action_fn(bc_rwr, DEVICE)
    policies["naive 离线 DQN"] = naive.act_fn()
    policies["CQL 离线 RL"] = cql.act_fn()

    rows = []
    for name, fn in policies.items():
        R, L, Dv = run_policy(mk_env(), fn, n_sessions=a.eval_sessions)
        rows.append((name, R, L, Dv))
    eval_path = ROOT / "experiments" / f"{cfg.run_name}_eval.csv"
    with open(eval_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["policy", "avg_reward", "avg_len", "genre_diversity"])
        for name, R, L, Dv in rows:
            w.writerow([name, round(R, 2), round(L, 2), round(Dv, 2)])
    print(f"\n[{cfg.run_name}] 策略对比（{a.eval_sessions} 会话）:")
    for name, R, L, Dv in sorted(rows, key=lambda x: -x[1]):
        print(f"  {name:<18} 奖励 {R:6.2f} | 长度 {L:5.1f} | 多样性 {Dv:.2f}")
    print(f"结果已写入 {eval_path.name}")


if __name__ == "__main__":
    main()
