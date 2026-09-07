# -*- coding: utf-8 -*-
"""
阶段 1：矩阵分解（Matrix Factorization）—— 推荐系统的"Hello World"
================================================================

【核心思想】
每个用户、每部电影都用一个 d 维向量（Embedding）表示。
用户对电影的喜欢程度 ≈ 两个向量的点积：

    预测评分 = 用户向量 · 电影向量 + 用户偏置 + 电影偏置 + 全局平均分

训练就是让预测评分逼近真实评分（MSE 损失），向量从随机初始化开始，
被梯度下降一点点"推"到合理的位置 —— 这就是 Embedding 的本质：
把离散的 ID 变成连续的、有语义的向量。

【运行方式】（在项目根目录）
    .venv/Scripts/python.exe src/train_mf.py --epochs 5
    想接着训练：.venv/Scripts/python.exe src/train_mf.py --epochs 5 --resume
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config import set_seed, ExperimentConfig

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "ml-1m"
CKPT_DIR = ROOT / "checkpoints"
EXP_DIR = ROOT / "experiments"
CKPT_DIR.mkdir(exist_ok=True)
EXP_DIR.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- 模型
class MatrixFactorization(nn.Module):
    """矩阵分解模型：Embedding 点积 + 偏置项。

    nn.Embedding(N, d) 本质就是一张 N×d 的查找表：
    输入 ID，输出对应的 d 维向量。它和普通参数一样参与梯度更新。
    """

    def __init__(self, n_users, n_items, dim):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.user_bias = nn.Embedding(n_users, 1)   # 有的用户就是打分偏高/偏低
        self.item_bias = nn.Embedding(n_items, 1)   # 有的电影就是普遍高分/低分
        # 小随机初始化：起点太大会导致训练初期梯度爆炸
        for emb in (self.user_emb, self.item_emb):
            nn.init.normal_(emb.weight, std=0.05)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user, item):
        dot = (self.user_emb(user) * self.item_emb(item)).sum(dim=-1)
        return (dot
                + self.user_bias(user).squeeze(-1)
                + self.item_bias(item).squeeze(-1))


# ---------------------------------------------------------------- 数据
def load_split():
    """加载数据并按"留一法"划分：每个用户时间最新的一条评分做测试集。

    这是推荐系统的标准评估协议（leave-one-out），模拟真实场景：
    用历史行为预测用户"下一次"会喜欢什么。
    """
    ratings = pd.read_csv(
        DATA / "ratings.dat", sep="::", engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )
    # 把原始 ID 映射为 0 开始的连续下标（Embedding 层要求）
    u_map = {u: i for i, u in enumerate(sorted(ratings["user_id"].unique()))}
    i_map = {m: i for i, m in enumerate(sorted(ratings["movie_id"].unique()))}
    ratings["u"] = ratings["user_id"].map(u_map)
    ratings["i"] = ratings["movie_id"].map(i_map)

    ratings = ratings.sort_values("timestamp")
    test = ratings.groupby("u").tail(1)      # 每人最新一条 → 测试
    train = ratings.drop(test.index)          # 其余 → 训练
    return train, test, u_map, i_map


# ---------------------------------------------------------------- 评估
@torch.no_grad()
def evaluate(model, train_seen, test, n_items, K=10):
    """全库排序评估：给每个用户的所有电影打分，看测试电影是否进 Top-K。

    - 先排除训练集里已经看过的电影（推荐系统不会重复推荐）
    - 只评估测试集中评分 ≥4 的用户：我们的目标是"预测用户下一部【喜欢】的
      电影"，他看了但打了 1 星的不该算推荐目标
    - Recall@K：测试电影进入 Top-K 的用户比例（命中率）
    - NDCG@K ：命中位置的折扣奖励，排越靠前得分越高
    """
    test = test[test["rating"] >= 4]
    model.eval()
    # 一次性算出 全用户 × 全电影 的得分矩阵（6040×3706，很小）
    scores = model.user_emb.weight @ model.item_emb.weight.T
    scores = scores + model.user_bias.weight + model.item_bias.weight.T
    for u, items in train_seen.items():
        scores[u, torch.as_tensor(items, device=scores.device)] = -1e9
    topk = scores.topk(K, dim=1).indices.cpu()   # [n_users, K]

    hits, ndcg = 0, 0.0
    test_by_user = test.groupby("u")["i"].first()
    for u, true_i in test_by_user.items():
        row = topk[u]
        if true_i in row.tolist():
            hits += 1
            rank = (row == true_i).nonzero().item()
            ndcg += 1.0 / np.log2(rank + 2)
    n = len(test_by_user)
    return hits / n, ndcg / n


# ---------------------------------------------------------------- 训练
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--dim", type=int, default=32, help="Embedding 维度")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=0.0,
                    help="weight decay 权重衰减（L2 正则化强度）")
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    ap.add_argument("--resume", action="store_true", help="从上次 checkpoint 继续训练")
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = ExperimentConfig.from_args(__file__, args, seed=args.seed)
    cfg.save(EXP_DIR)
    print(f"实验配置已保存: {cfg.run_name}")

    train, test, u_map, i_map = load_split()
    n_users, n_items = len(u_map), len(i_map)
    print(f"设备: {DEVICE} | 用户 {n_users} 物品 {n_items} "
          f"| 训练 {len(train)} 条 测试 {len(test)} 条")

    model = MatrixFactorization(n_users, n_items, args.dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = nn.MSELoss()

    start_epoch = 0
    ckpt_path = CKPT_DIR / "mf.pt"
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=True)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt["epoch"]
        print(f"已从 checkpoint 恢复（epoch {start_epoch}），继续训练")

    # 训练集张量 + 每个用户看过的物品集合（评估时排除用）
    u_all = torch.as_tensor(train["u"].values, device=DEVICE)
    i_all = torch.as_tensor(train["i"].values, device=DEVICE)
    r_all = torch.as_tensor(train["rating"].values, dtype=torch.float32,
                            device=DEVICE)
    train_seen = train.groupby("u")["i"].apply(list).to_dict()

    log_path = EXP_DIR / "mf_log.csv"
    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        model.train()
        t0, total, n_batch = time.time(), 0.0, 0
        perm = torch.randperm(len(u_all), device=DEVICE)  # 每轮打乱顺序
        for s in range(0, len(perm), args.batch):
            idx = perm[s:s + args.batch]
            pred = model(u_all[idx], i_all[idx])
            loss = loss_fn(pred, r_all[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            n_batch += 1

        recall, ndcg = evaluate(model, train_seen, test, n_items)
        secs = time.time() - t0
        print(f"epoch {epoch:2d} | train_loss {total / n_batch:.4f} "
              f"| Recall@10 {recall:.4f} | NDCG@10 {ndcg:.4f} | {secs:.1f}s")

        # 追加实验日志（实验记录是工程师的基本功！）
        # 第一列是实验标签（超参数组合），方便多次实验对比
        run = f"dim{args.dim}-lr{args.lr}-wd{args.wd}"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{run},{datetime.now():%Y-%m-%d %H:%M:%S},{epoch},"
                    f"{total / n_batch:.6f},{recall:.6f},{ndcg:.6f},{secs:.1f}\n")
        # 每个 epoch 存一次 checkpoint，随时可以断点续跑
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "dim": args.dim}, ckpt_path)

    show_samples(model, test, train_seen, i_map)


def show_samples(model, test, train_seen, i_map, n_show=3):
    """挑几个用户，展示模型给出的 Top-10 推荐（最直观的产出）。"""
    movies = pd.read_csv(
        DATA / "movies.dat", sep="::", engine="python",
        names=["movie_id", "title", "genres"], encoding="latin-1",
    ).set_index("movie_id")
    idx2movie = {v: k for k, v in i_map.items()}

    with torch.no_grad():
        scores = model.user_emb.weight @ model.item_emb.weight.T
        scores = scores + model.user_bias.weight + model.item_bias.weight.T
        for u, items in train_seen.items():
            scores[u, torch.as_tensor(items, device=scores.device)] = -1e9
        topk = scores.topk(10, dim=1).indices.cpu()

    out = ["# 矩阵分解模型 · 推荐效果抽样\n"]
    rng = np.random.default_rng(0)
    for u in rng.choice(sorted(train_seen), size=n_show, replace=False):
        out.append(f"## 用户 {u}\n")
        out.append("推荐 Top-10：")
        for rank, i in enumerate(topk[u].tolist(), 1):
            title = movies.loc[idx2movie[i], "title"]
            genres = movies.loc[idx2movie[i], "genres"]
            out.append(f"{rank}. {title}  （{genres}）")
        out.append("")
    path = EXP_DIR / "01-矩阵分解-推荐样例.md"
    path.write_text("\n".join(out), encoding="utf-8")
    print("\n推荐样例已写入:", path)


if __name__ == "__main__":
    main()
