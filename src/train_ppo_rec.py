# -*- coding: utf-8 -*-
"""
Phase 2 · 模块 2（PPO 对照）：手写 Proximal Policy Optimization 做序列推荐
============================================================================

【和 REINFORCE + Actor-Critic 的关系】
`train_pg_rec.py` 的 PG 是 PPO 家族的祖先：MC 回报 G_t 当目标、V(s) 当基线，
每批数据只用一次。PPO 在其上加了三个工程旋钮（Schulman et al. 2017,
arXiv:1707.06347）：

1. **GAE(λ) 优势估计**：λ=0.95 用"一步 TD + 后续衰减"的折中，比 MC(λ=1)
   方差小、比单步 TD 偏差小；
2. **重要性采样比 + 裁剪**：旧数据可复用多轮，但 ratio 超出 [1-ε, 1+ε]
   就截断，保证每次更新"只改一点"，消除 PG 的大步长发散风险；
3. **多 epoch 小批量**：一份 rollout 复用 K 轮，样本效率高于 PG。

GRPO（DeepSeekMath/R1，2024-2025）是 PPO 的"去价值网络"变体：用同组候选的
相对奖励代替 Critic，显存更省——见 `notes/14-前沿RL论文与落地路线.md`。

【运行方式】
.venv/Scripts/python.exe src/train_ppo_rec.py --updates 40
.venv/Scripts/python.exe src/train_ppo_rec.py --updates 1 --no_eval   # 冒烟
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
from train_pg_rec import ActorCritic

ROOT = Path(__file__).resolve().parent.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """GAE(λ) 优势估计，返回 ``(advantages, returns)``。

    A_t = δ_t + γλ·A_{t+1}，其中 δ_t = r_t + γ·V(s_{t+1})·(1-done) − V(s_t)。
    λ=1 且 V≡0 时退化为蒙特卡洛折扣回报（与 PG 的 G_t 一致）。
    """
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float32)
    returns = np.zeros(n, dtype=np.float32)
    gae_ = 0.0
    next_v = 0.0
    for t in reversed(range(n)):
        delta = rewards[t] + gamma * next_v * (1 - dones[t]) - values[t]
        gae_ = delta + gamma * lam * (1 - dones[t]) * gae_
        adv[t] = gae_
        returns[t] = gae_ + values[t]
        next_v = values[t]
    return adv, returns


def collect_rollouts(ac, env, episodes, gamma, lam):
    """用当前策略采样 episodes，返回 (S, A, logp_old, returns) 与平均回报。"""
    batch_s, batch_a, batch_logp, batch_ret = [], [], [], []
    rets = []
    for _ in range(episodes):
        s = env.reset()
        traj_s, traj_a, traj_logp, traj_r, traj_d, traj_v = (
            [], [], [], [], [], [])
        R = 0.0
        while True:
            st = torch.from_numpy(s).float().unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits, v = ac(st)
                dist = torch.distributions.Categorical(logits=logits)
                a = int(dist.sample())
                logp = float(dist.log_prob(torch.tensor(a, device=DEVICE)))
            s2, r, d, _ = env.step(a)
            R += r
            traj_s.append(s)
            traj_a.append(a)
            traj_logp.append(logp)
            traj_r.append(r)
            traj_d.append(d)
            traj_v.append(float(v.item()))
            s = s2
            if d:
                break
        rets.append(R)
        _, returns = compute_gae(
            np.asarray(traj_r, dtype=np.float32),
            np.asarray(traj_v, dtype=np.float32),
            np.asarray(traj_d, dtype=np.float32),
            gamma, lam,
        )
        batch_s.extend(traj_s)
        batch_a.extend(traj_a)
        batch_logp.extend(traj_logp)
        batch_ret.extend(returns.tolist())
    return (
        np.stack(batch_s),
        np.asarray(batch_a, dtype=np.int64),
        np.asarray(batch_logp, dtype=np.float32),
        np.asarray(batch_ret, dtype=np.float32),
    ), rets


def ppo_update(ac, opt, S, A, logp_old, returns, adv,
               clip, epochs, mb, ent):
    """PPO 核心更新：clip 后的 surrogate + 价值损失 + 熵正则。"""
    n = len(S)
    order = np.arange(n)
    total_policy = total_value = total_ent = 0.0
    steps = 0
    for _ in range(epochs):
        np.random.shuffle(order)
        for start in range(0, n, mb):
            idx = order[start:start + mb]
            Sb = torch.from_numpy(S[idx]).float().to(DEVICE)
            Ab = torch.as_tensor(A[idx], device=DEVICE)
            logp_old_b = torch.as_tensor(
                logp_old[idx], device=DEVICE)
            ret_b = torch.as_tensor(returns[idx], device=DEVICE)
            adv_b = torch.as_tensor(adv[idx], device=DEVICE)

            logits, v = ac(Sb)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(Ab)
            ratio = (logp - logp_old_b).exp()
            surr1 = ratio * adv_b
            surr2 = ratio.clamp(1.0 - clip, 1.0 + clip) * adv_b
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * F.mse_loss(v, ret_b)
            entropy = dist.entropy().mean()
            loss = policy_loss + value_loss - ent * entropy

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ac.parameters(), 10.0)
            opt.step()
            total_policy += policy_loss.item()
            total_value += value_loss.item()
            total_ent += entropy.item()
            steps += 1
    return (total_policy / steps, total_value / steps, total_ent / steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=40)
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=4,
                    help="每份 rollout 复用几轮（PPO 的 K）")
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--lam", type=float, default=0.95,
                    help="GAE λ；1.0 退化为蒙特卡洛回报")
    ap.add_argument("--mb", type=int, default=128, help="小批量大小")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_eval", action="store_true")
    ap.add_argument("--ckpt", default="ppo_rec.pt")
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = ExperimentConfig.from_args(__file__, args, seed=args.seed)
    cfg.save(ROOT / "experiments")
    print(f"实验配置已保存: {cfg.run_name}")

    sasrec, maxlen, _ = load_user_model(
        str(ROOT / "checkpoints" / "sasrec_main.pt"))
    assets = load_rec_assets()
    ac = ActorCritic().to(DEVICE)
    opt = torch.optim.Adam(ac.parameters(), lr=args.lr)
    ckpt_path = ROOT / "checkpoints" / args.ckpt

    start = 1
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        ac.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = ck["update"] + 1
        print(f"[resume] 从 update {start} 继续", flush=True)

    env = RecSimEnv(sasrec, maxlen, assets)
    log_path = ROOT / "experiments" / "ppo_rec_log.csv"
    if not args.resume or not log_path.exists():
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["update", "avg_return", "policy_loss", "value_loss",
                 "entropy", "elapsed_min"])

    print(f"设备 {DEVICE} | update {start}~{start + args.updates - 1} | "
          f"{args.episodes} 回合/更新 | K={args.epochs} clip={args.clip} "
          f"λ={args.lam}", flush=True)
    t0 = time.time()
    for upd in range(start, start + args.updates):
        (S, A, logp_old, returns), rets = collect_rollouts(
            ac, env, args.episodes, GAMMA, args.lam)
        with torch.no_grad():
            _, v_old = ac(torch.from_numpy(S).float().to(DEVICE))
        adv = (returns - v_old.cpu().numpy()).astype(np.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)   # 标准化优势
        p_loss, v_loss, ent = ppo_update(
            ac, opt, S, A, logp_old, returns, adv,
            args.clip, args.epochs, args.mb, args.ent)

        if upd % 5 == 0:
            el = (time.time() - t0) / 60
            print(f"update {upd:3d} | 平均回合奖励 {np.mean(rets):6.2f} "
                  f"| 熵 {ent:.2f} | {el:.1f}min", flush=True)
            with open(log_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    upd, round(float(np.mean(rets)), 4),
                    round(p_loss, 4), round(v_loss, 4),
                    round(ent, 4), round(el, 2)])
        torch.save({"model": ac.state_dict(), "opt": opt.state_dict(),
                    "update": upd}, ckpt_path)

    if args.no_eval:
        print("跳过评估(--no_eval)")
        return

    def ppo_act(e, s):
        with torch.no_grad():
            logits, _ = ac(torch.from_numpy(s).float()
                           .unsqueeze(0).to(DEVICE))
        return int(logits.argmax(1))

    def ppo_sample(e, s):
        with torch.no_grad():
            logits, _ = ac(torch.from_numpy(s).float()
                           .unsqueeze(0).to(DEVICE))
        return int(torch.distributions.Categorical(
            logits=logits).sample())

    mk = lambda: RecSimEnv(sasrec, maxlen, assets)
    rows = [("随机", *run_policy(mk(), lambda e, s: e.rng.randrange(N_ACTIONS))),
            ("贪心(短视)", *run_policy(mk(), lambda e, s: greedy_action(e))),
            ("PPO(argmax)", *run_policy(mk(), ppo_act)),
            ("PPO(采样)", *run_policy(mk(), ppo_sample))]
    with open(ROOT / "experiments" / "ppo_rec_eval.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["policy", "avg_reward", "avg_len", "genre_diversity"])
        for name, R, L, D in rows:
            w.writerow([name, round(R, 2), round(L, 2), round(D, 2)])
    print("========== 300 个评估会话（相同种子，与 PG/DQN 同一协议）==========")
    for name, R, L, D in rows:
        print(f"  {name}: 奖励 {R:.2f} | 长度 {L:.1f} | 多样性 {D:.2f}")


if __name__ == "__main__":
    main()
