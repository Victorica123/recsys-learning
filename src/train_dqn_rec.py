# -*- coding: utf-8 -*-
"""模块2 v2: 推荐会话 MDP 环境(修复版) + 稳定化 Double DQN。

v1 发散诊断(Q 值从合理值 ~2.5 爆炸到 549):
  1) Polyak 软更新让在线/目标网络高度相关 → Double DQN 的去偏机制失效
  2) 单步 4 次梯度更新 × 重复回放 → max 过估计被反复放大
  3) 环境动力学太弱: 疲劳按"会话累计次数"惩罚, 贪心策略几乎不受惩罚,
     可学习信号弱, 噪声主导了 bootstrap

v2 修复:
  环境:
  - 疲劳按"连续同类型次数"惩罚 (0.75^n), 换类型即恢复 → 交错推荐是最优解,
    贪心(无视状态)必然持续衰减 → 创造出 RL 可学习的长期优势
  - 用户厌倦退出: 连续 3 次无感推荐 → 会话提前终止(失去未来所有奖励)
  - 状态追加"连续无感计数"维度
  DQN:
  - LayerNorm 稳定价值函数; lr 1e-4; 硬目标同步(每500次更新)使 Double 去偏生效
  - 每环境步仅 1 次梯度更新; batch 256; Huber loss
"""
import os, csv, math, time, random, argparse
from collections import deque
from pathlib import Path

import numpy as np

from config import set_seed, ExperimentConfig
from constants import GENRES
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from train_sasrec import SASRec, load_sequences

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "ml-1m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GAMMA = 0.9
SESSION = 20
N_ACTIONS = 200
STATE_DIM = 64 + len(GENRES) + 1   # ctx + 最近5次推荐类型直方图 + 连续无感计数


def load_user_model(ckpt_path):
    """加载冻结的 SASRec 用户模型, 返回 (model, maxlen, n_items)。"""
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    a = ck["args"]
    n_items = ck["model"]["item_emb.weight"].shape[0] - 1
    model = SASRec(n_items=n_items, d=a["d"], max_len=a["maxlen"],
                   n_blocks=a["blocks"], dropout=a["dropout"]).to(DEVICE)
    model.load_state_dict(ck["model"]); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, a["maxlen"], n_items


def load_rec_assets():
    """Top200 热门物品(动作空间)、物品→首类型下标、训练序列(与SASRec同切分)。"""
    ratings = pd.read_csv(DATA / "ratings.dat", sep="::", engine="python",
                          names=["user_id", "movie_id", "rating", "timestamp"])
    movies = pd.read_csv(DATA / "movies.dat", sep="::", engine="python",
                         encoding="latin-1",
                         names=["movie_id", "title", "genres"])
    i_map = {m: i + 1 for i, m in enumerate(sorted(ratings["movie_id"].unique()))}
    top_movies = ratings["movie_id"].value_counts().head(N_ACTIONS).index.tolist()
    top_items = np.array([i_map[m] for m in top_movies], dtype=np.int64)
    g2i = {g: i for i, g in enumerate(GENRES)}
    m2g = movies.set_index("movie_id")["genres"].to_dict()
    item_genre = {i: g2i[m2g[m].split("|")[0]] for m, i in i_map.items()}
    seqs, _ = load_sequences()
    train_seq = [s[:-2] for s in seqs if len(s) > 7]   # 末两位留给 valid/test
    return top_items, item_genre, train_seq


class RecSimEnv:
    """SASRec 当用户模型; 动作 = Top200 热门物品; 20 步会话。"""

    def __init__(self, sasrec, maxlen, assets, seed=0):
        self.sasrec, self.maxlen = sasrec, maxlen
        self.top_items, self.genre, self.train_seq = assets
        self.rng = random.Random(seed)

    def _score(self, hist):
        L = min(len(hist), self.maxlen)
        seq = torch.tensor([hist[-L:]], device=DEVICE)
        items = torch.from_numpy(self.top_items).to(DEVICE)
        with torch.no_grad():
            ctx = self.sasrec.encode(seq)[0, -1]
            s = self.sasrec.score(ctx, items)
        return ctx.cpu().numpy(), s.cpu().numpy()

    def _state(self, ctx):
        hist = np.zeros(len(GENRES), dtype=np.float32)
        for it in self.recent[-5:]:
            hist[self.genre.get(int(it), 0)] += 1.0
        return np.concatenate([ctx, hist / 5.0,
                               np.array([self.consec_dislike / 3.0],
                                        dtype=np.float32)])

    def reset(self):
        u = self.rng.randrange(len(self.train_seq))
        seq = self.train_seq[u]
        k = self.rng.randint(5, min(15, len(seq) - 1))
        self.hist = list(seq[:k]); self.u = u
        self.recent = []
        self.last_genre, self.run_len = -1, 0
        self.consec_dislike = 0
        ctx, _ = self._score(self.hist)
        return self._state(ctx)

    def step(self, a):
        item = int(self.top_items[a]); g = self.genre.get(item, 0)
        if g == self.last_genre:
            self.run_len += 1
        else:
            self.last_genre, self.run_len = g, 1
        ctx, scores = self._score(self.hist)
        p = 1.0 / (1.0 + math.exp(-float(scores[a])))
        p *= 0.75 ** (self.run_len - 1)          # 连续同类型疲劳, 换类型恢复
        like = self.rng.random() < p
        self.recent.append(item)
        if like:
            self.hist.append(item)
            self.consec_dislike = 0
            r = 0.9
        else:
            self.consec_dislike += 1
            r = -0.1
        done = (len(self.recent) >= SESSION) or (self.consec_dislike >= 3)  # 厌倦退出
        return self._state(ctx), r, done, {"p": p, "genre": g, "like": like}


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, N_ACTIONS))

    def forward(self, x):
        return self.net(x)


def greedy_action(env):
    """短视贪心: 每步用当前历史重新打分, 选即时 p 最大的物品(无视疲劳与厌倦)。"""
    _, scores = env._score(env.hist)
    return int(np.argmax(scores))


def greedy_varied_action(env):
    """贪心去重(换类型)基线: 仍短视选最高分, 但跳过与上一步相同类型。

    原贪心每步都推同一部高分电影, 撞 0.75^n 连续疲劳衰减, 奖励塌缩更像
    "策略愚蠢"而非"长期规划有价值"。本基线换类型避开疲劳, 是更公平的对手。
    """
    _, scores = env._score(env.hist)
    order = np.argsort(-scores)
    for a in order:
        g = env.genre.get(int(env.top_items[a]), 0)
        if env.last_genre < 0 or g != env.last_genre:
            return int(a)
    return int(order[0])


def run_policy(env, policy_fn, n_sessions=300):
    """通用评估: policy_fn(env, state) -> action。返回 (平均奖励, 平均长度, 类型多样性)。"""
    rs, ls, div = [], [], []
    for _ in range(n_sessions):
        s = env.reset(); R, L, genres = 0.0, 0, []
        while True:
            a = policy_fn(env, s)
            s, r, d, info = env.step(a)
            R += r; L += 1; genres.append(info["genre"])
            if d:
                break
        rs.append(R); ls.append(L); div.append(len(set(genres)))
    return float(np.mean(rs)), float(np.mean(ls)), float(np.mean(div))


def evaluate(qnet, env, n_sessions=150):
    def act(e, s):
        with torch.no_grad():
            return int(qnet(torch.from_numpy(s).float()
                            .unsqueeze(0).to(DEVICE)).argmax(1))
    return run_policy(env, act, n_sessions)


def train(ckpt, steps, lr, eps_end, tag, resume=None):
    sasrec, maxlen, _ = load_user_model(ckpt)
    assets = load_rec_assets()
    env = RecSimEnv(sasrec, maxlen, assets)
    q = QNet().to(DEVICE); qt = QNet().to(DEVICE)
    qt.load_state_dict(q.state_dict())
    opt = optim.Adam(q.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()
    buf = deque(maxlen=100_000)
    start_eps = 1.0
    if resume and os.path.exists(resume):
        ck = torch.load(resume, map_location=DEVICE, weights_only=True)
        q.load_state_dict(ck["model"]); qt.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"]); start_eps = ck["eps"]
        print(f"[resume] {resume} eps={start_eps:.3f}", flush=True)

    log_path = f"experiments/dqn_{tag}_log.csv"
    new_file = not os.path.exists(log_path) or resume is None
    f = open(log_path, "a", newline="", encoding="utf-8"); w = None
    s = env.reset()
    t0, n_upd = time.time(), 0
    for step in range(1, steps + 1):
        eps = max(eps_end, start_eps - (start_eps - eps_end) * (step / (0.6 * steps)))
        if random.random() < eps:
            a = random.randrange(N_ACTIONS)
        else:
            with torch.no_grad():
                a = int(q(torch.from_numpy(s).float().unsqueeze(0).to(DEVICE)).argmax(1))
        s2, r, d, _ = env.step(a)
        buf.append((s, a, r, s2, d)); s = s2
        if d:
            s = env.reset()
        if len(buf) >= 2000:
            batch = random.sample(buf, 256)
            S = torch.from_numpy(np.array([b[0] for b in batch])).float().to(DEVICE)
            A = torch.tensor([b[1] for b in batch]).long().to(DEVICE).unsqueeze(1)
            R = torch.tensor([b[2] for b in batch]).float().to(DEVICE).unsqueeze(1)
            S2 = torch.from_numpy(np.array([b[3] for b in batch])).float().to(DEVICE)
            D = torch.tensor([b[4] for b in batch]).float().to(DEVICE).unsqueeze(1)
            qv = q(S).gather(1, A)
            with torch.no_grad():
                a_star = q(S2).argmax(1, keepdim=True)          # Double DQN: 在线网选动作
                y = R + GAMMA * (1 - D) * qt(S2).gather(1, a_star)  # 目标网估值(硬同步, 去相关)
            loss = loss_fn(qv, y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10.0); opt.step()
            n_upd += 1
            if n_upd % 500 == 0:
                qt.load_state_dict(q.state_dict())              # 硬目标同步
        if step % 1000 == 0:
            with torch.no_grad():
                qb = torch.from_numpy(np.array(
                    [b[0] for b in random.sample(buf, min(256, len(buf)))])).float().to(DEVICE)
                qmean = q(qb).mean().item()
            row = {"step": step, "eps": round(eps, 3), "q_mean": round(qmean, 3),
                   "elapsed_min": round((time.time() - t0) / 60, 1)}
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if new_file:
                    w.writeheader()
            w.writerow(row); f.flush()
            print(f"step {step}/{steps} eps={eps:.2f} Q均值={qmean:.2f}", flush=True)
        if step % 3000 == 0:
            Rm, Lm, Dm = evaluate(q, env, 100)
            print(f"  [eval@{step}] 会话奖励 {Rm:.2f} 长度 {Lm:.1f} 多样性 {Dm:.2f}",
                  flush=True)
            torch.save({"model": q.state_dict(), "opt": opt.state_dict(), "eps": eps},
                       f"checkpoints/dqn_{tag}_last.pt")
    f.close()
    torch.save({"model": q.state_dict(), "opt": opt.state_dict(), "eps": eps_end},
               f"checkpoints/dqn_{tag}_last.pt")

    final_compare(sasrec, maxlen, assets, q, tag)


def final_compare(sasrec, maxlen, assets, q, tag, n_sessions=300):
    """四方策略对比: 随机 / 短视贪心 / 贪心去重(换类型) / Double DQN。

    每个策略用相同种子新建环境 → 面对完全相同的会话流, 保证公平。
    """
    mk = lambda: RecSimEnv(sasrec, maxlen, assets)

    def dqn_act(e, s):
        with torch.no_grad():
            return int(q(torch.from_numpy(s).float().unsqueeze(0).to(DEVICE)).argmax(1))

    rows = [("随机策略", *run_policy(mk(), lambda e, s: e.rng.randrange(N_ACTIONS), n_sessions)),
            ("贪心策略(即时最优)", *run_policy(mk(), lambda e, s: greedy_action(e), n_sessions)),
            ("贪心去重(换类型)", *run_policy(mk(), lambda e, s: greedy_varied_action(e), n_sessions)),
            ("Double DQN", *run_policy(mk(), dqn_act, n_sessions))]
    with open(f"experiments/dqn_{tag}_eval.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["policy", "avg_reward", "avg_len", "genre_diversity"])
        for name, R, L, Dv in rows:
            w.writerow([name, round(R, 2), round(L, 2), round(Dv, 2)])
    print(f"[{tag}] 最终对比({n_sessions}会话):")
    for name, R, L, Dv in rows:
        print(f"  {name}: 奖励 {R:.2f} 长度 {L:.1f} 多样性 {Dv:.2f}")


def compare_only(ckpt, tag, resume, n_sessions=300):
    """只评估不训练: 加载已训练 DQN 权重, 跑四方对比(含贪心去重基线)。"""
    sasrec, maxlen, _ = load_user_model(ckpt)
    assets = load_rec_assets()
    q = QNet().to(DEVICE)
    ck = torch.load(resume, map_location=DEVICE, weights_only=True)
    q.load_state_dict(ck["model"])
    q.eval()
    final_compare(sasrec, maxlen, assets, q, tag, n_sessions)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    ap.add_argument("--ckpt", default="checkpoints/sasrec_main.pt")
    ap.add_argument("--steps", type=int, default=7000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eps_end", type=float, default=0.05)
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--eval-only", action="store_true",
                    help="只加载已训练 DQN 跑四方对比(含贪心去重基线), 不训练")
    a = ap.parse_args()
    set_seed(a.seed)
    cfg = ExperimentConfig.from_args(__file__, a, seed=a.seed)
    cfg.save(ROOT / "experiments")
    print(f"实验配置已保存: {cfg.run_name}")
    if a.eval_only:
        resume = a.resume or f"checkpoints/dqn_{a.tag}_last.pt"
        compare_only(a.ckpt, a.tag, resume)
    else:
        train(a.ckpt, a.steps, a.lr, a.eps_end, a.tag, a.resume)
