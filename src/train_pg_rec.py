# -*- coding: utf-8 -*-
"""
Phase 2 · 模块 2（对照方案）：REINFORCE + Actor-Critic 策略梯度做序列推荐
===========================================================================

【为什么有这个对照方案】
同环境的手写 DQN v1 反复发散（Q 值爆炸到 500+）——这正是 Sutton 说的
"致命三要素"：函数逼近 + 自举（bootstrap）+ 离策略采样同时存在时，
值函数可能无界增长。max 操作对 200 个动作的噪声 Q 值取最大值，
每次备份都注入正偏差，Polyak/Double DQN/梯度裁剪都只能缓解不能根治。

策略梯度走了完全不同的路：**不对未来做自举**，直接采样完整回合的
真实回报 G_t，用 (G_t − baseline) 作为优势估计，沿 log π(a|s) 上升梯度。
蒙特卡洛估计无偏、没有自举，天然不会发散 —— 代价是方差大，所以用
学习的价值基线（Actor-Critic）来压方差。这也是 PPO/GRPO 家族的祖先:
PPO = Actor-Critic + 重要性采样比裁剪 + 多轮复用数据。

【运行方式】
.venv/Scripts/python.exe src/train_pg_rec.py --updates 40
.venv/Scripts/python.exe src/train_pg_rec.py --updates 40 --resume   # 断点续训
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import set_seed, ExperimentConfig

from train_dqn_rec import (RecSimEnv, load_user_model, load_rec_assets,
                           run_policy, greedy_action,
                           N_ACTIONS, GAMMA, STATE_DIM)

ROOT = Path(__file__).resolve().parent.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = ROOT / "checkpoints" / "pg_rec.pt"


class ActorCritic(nn.Module):
    """共享主干: Actor 输出 200 个动作的 logits; Critic 输出 V(s) 基线。"""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(STATE_DIM, 128), nn.ReLU())
        self.actor = nn.Sequential(nn.Linear(128, 128), nn.ReLU(),
                                   nn.Linear(128, N_ACTIONS))
        self.critic = nn.Sequential(nn.Linear(128, 64), nn.ReLU(),
                                    nn.Linear(64, 1))

    def forward(self, s):
        h = self.backbone(s)
        return self.actor(h), self.critic(h).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=40)
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    ap.add_argument("--no_eval", action="store_true",
                    help="跳过结尾的三方评估(中间续训段用)")
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = ExperimentConfig.from_args(__file__, args, seed=args.seed)
    cfg.save(ROOT / "experiments")
    print(f"实验配置已保存: {cfg.run_name}")
    sasrec, maxlen, _ = load_user_model(str(ROOT / "checkpoints" / "sasrec_main.pt"))
    assets = load_rec_assets()
    ac = ActorCritic().to(DEVICE)
    opt = torch.optim.Adam(ac.parameters(), lr=args.lr)

    start = 1
    if args.resume and CKPT.exists():
        ck = torch.load(CKPT, map_location=DEVICE, weights_only=True)
        ac.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start = ck["update"] + 1
        print(f"[resume] 从 update {start} 继续", flush=True)

    env = RecSimEnv(sasrec, maxlen, assets)   # 单一 env 复用, rng 跨回合持续演化
    log_path = ROOT / "experiments" / "pg_log.csv"
    if not args.resume or not log_path.exists():
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["update", "avg_return", "entropy",
                                    "elapsed_min"])

    print(f"设备 {DEVICE} | update {start}~{start + args.updates - 1} | "
          f"{args.episodes} 回合/更新", flush=True)
    t0 = time.time()
    for upd in range(start, start + args.updates):
        # ---------- 用当前策略采样一批回合 ----------
        batch_s, batch_a, batch_G, rets = [], [], [], []
        for _ in range(args.episodes):
            s = env.reset()
            traj, R = [], 0.0
            while True:
                st = torch.from_numpy(s).float()
                with torch.no_grad():
                    logits, _ = ac(st.unsqueeze(0).to(DEVICE))
                a = int(torch.distributions.Categorical(logits=logits).sample())
                s, r, d, _ = env.step(a)
                traj.append((st, a, r)); R += r
                if d:
                    break
            rets.append(R)
            G = 0.0                                  # 从后往前算折扣回报
            for st, a, r in reversed(traj):
                G = r + GAMMA * G
                batch_s.append(st); batch_a.append(a); batch_G.append(G)

        # ---------- 一次 Actor-Critic 更新 ----------
        S = torch.stack(batch_s).to(DEVICE)
        A = torch.as_tensor(batch_a, device=DEVICE)
        Gb = torch.as_tensor(batch_G, dtype=torch.float32, device=DEVICE)
        logits, v = ac(S)
        adv = Gb - v.detach()                  # 优势 = 实际回报 − 基线
        logp = F.log_softmax(logits, dim=-1).gather(1, A.unsqueeze(1)).squeeze(1)
        ent = -(F.softmax(logits, -1) *
                F.log_softmax(logits, -1)).sum(-1).mean()
        loss = -(logp * adv).mean() + 0.5 * F.mse_loss(v, Gb) - args.ent * ent
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(ac.parameters(), 10.0)
        opt.step()

        if upd % 5 == 0:
            el = (time.time() - t0) / 60
            print(f"update {upd:3d} | 平均回合奖励 {np.mean(rets):6.2f} "
                  f"| 熵 {ent.item():.2f} | {el:.1f}min", flush=True)
            with open(log_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([upd, round(float(np.mean(rets)), 4),
                                        round(ent.item(), 4), round(el, 2)])
        torch.save({"model": ac.state_dict(), "opt": opt.state_dict(),
                    "update": upd}, CKPT)

    # ---------- 公平对决: 三策略面对相同种子的会话流 ----------
    if args.no_eval:
        print("跳过评估(--no_eval)")
        return

    def pg_act(e, s):
        with torch.no_grad():
            logits, _ = ac(torch.from_numpy(s).float().unsqueeze(0).to(DEVICE))
        return int(logits.argmax(1))

    def pg_sample(e, s):
        with torch.no_grad():
            logits, _ = ac(torch.from_numpy(s).float().unsqueeze(0).to(DEVICE))
        return int(torch.distributions.Categorical(logits=logits).sample())

    mk = lambda: RecSimEnv(sasrec, maxlen, assets)
    rows = [("随机", *run_policy(mk(), lambda e, s: e.rng.randrange(N_ACTIONS))),
            ("贪心(短视)", *run_policy(mk(), lambda e, s: greedy_action(e))),
            ("策略梯度(argmax)", *run_policy(mk(), pg_act)),
            ("策略梯度(采样)", *run_policy(mk(), pg_sample))]
    with open(ROOT / "experiments" / "pg_eval.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["policy", "avg_reward", "avg_len", "genre_diversity"])
        for name, R, L, D in rows:
            w.writerow([name, round(R, 2), round(L, 2), round(D, 2)])
    print("========== 300 个评估会话（相同种子，公平对比）==========")
    for name, R, L, D in rows:
        print(f"  {name}: 奖励 {R:.2f} | 长度 {L:.1f} | 多样性 {D:.2f}")


if __name__ == "__main__":
    main()
