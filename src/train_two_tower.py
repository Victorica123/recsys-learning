# -*- coding: utf-8 -*-
"""
阶段 3：双塔召回模型 + Faiss 向量检索
======================================

【为什么需要双塔？】
阶段 2 的 DeepFM 是"排序模型"：给它一个 (用户, 电影) 对，它打一个分。
但真实业务有几百万物品，不可能每个用户都把全库打一遍分。于是工业界用
两段式架构：
    召回（双塔）：用户塔和物品塔分别算向量，Faiss 里毫秒级找出最相近的
                 几百个候选 —— 快，但物品和用户没见过面，精度有限
    排序（DeepFM）：只对这几百个候选精排 —— 慢但准
双塔是"召回"的工业标准答案（YouTube 2019 论文带火，至今各家都在用）。

【训练方式：批内负采样 softmax】
一个 batch 里 B 个 (用户, 正样本电影)。用户塔出 B 个用户向量，
物品塔出 B 个电影向量，两两做点积得到 B×B 打分矩阵 —— 对角线是正样本，
其余 B-1 个就是"免费的负样本"，用交叉熵逼着正样本得分最高。

【运行方式】
    .venv/Scripts/python.exe src/train_two_tower.py --epochs 20
"""
import argparse
import time
from pathlib import Path

import numpy as np

from config import set_seed, ExperimentConfig
from constants import GENRES
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "ml-1m"
CKPT_DIR = ROOT / "checkpoints"
EXP_DIR = ROOT / "experiments"
CKPT_DIR.mkdir(exist_ok=True)
EXP_DIR.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEMP = 0.05  # softmax 温度：越小分布越尖锐，对比学习常用技巧


# ---------------------------------------------------------------- 数据
def load_data():
    """隐式反馈视角：评分≥4 记为一次'喜欢'，每人最新的一次喜欢做测试。"""
    ratings = pd.read_csv(DATA / "ratings.dat", sep="::", engine="python",
                          names=["user_id", "movie_id", "rating", "timestamp"])
    users = pd.read_csv(DATA / "users.dat", sep="::", engine="python",
                        names=["user_id", "gender", "age", "occupation", "zip"])
    movies = pd.read_csv(DATA / "movies.dat", sep="::", engine="python",
                         names=["movie_id", "title", "genres"],
                         encoding="latin-1")

    u_map = {u: i for i, u in enumerate(sorted(ratings["user_id"].unique()))}
    i_map = {m: i for i, m in enumerate(sorted(ratings["movie_id"].unique()))}
    ratings["u"] = ratings["user_id"].map(u_map)
    ratings["i"] = ratings["movie_id"].map(i_map)

    pos = ratings[ratings["rating"] >= 4]
    # 与阶段 1 完全相同的评估协议：每人最新一条【评分】做测试，
    # 只保留其中 ≥4 的用户（预测"下一部喜欢的电影"）
    last = ratings.sort_values("timestamp").groupby("u").tail(1)
    test = last[last["rating"] >= 4]
    train = pos.drop(test.index)

    # 用户属性表（按用户下标对齐，塔里要查）
    users = users.set_index("user_id").loc[
        sorted(ratings["user_id"].unique())].reset_index()
    # 注意：age 原始值是 1/18/25/35/45/50/56 七个桶，必须重新映射成 0~6
    age_map = {v: k for k, v in enumerate(sorted(users["age"].unique()))}
    u_attr = np.stack([
        users["user_id"].map(u_map).values,
        (users["gender"] == "M").astype(int).values,
        users["age"].map(age_map).values,
        users["occupation"].values], axis=1)

    # 物品的类型 multi-hot（padding 到下标 len(GENRES)）
    g2i = {g: k for k, g in enumerate(GENRES)}
    n_items = len(i_map)
    gid = np.full((n_items, 6), len(GENRES), dtype=np.int64)
    gmask = np.zeros((n_items, 6), dtype=np.float32)
    movies = movies.set_index("movie_id")
    for m, i in i_map.items():
        gs = [g2i[g] for g in movies.loc[m, "genres"].split("|")]
        gid[i, :len(gs)] = gs
        gmask[i, :len(gs)] = 1.0

    maps = {"u_map": {int(k): v for k, v in u_map.items()},
            "i_map": {int(k): v for k, v in i_map.items()}, "g2i": g2i}
    return train, test, u_attr, gid, gmask, maps


# ---------------------------------------------------------------- 模型
class Tower(nn.Module):
    """一个塔 = 若干 Embedding 查表拼接 + 两层 MLP 压缩成统一维度。"""

    def __init__(self, sizes, dims, in_extra, out_dim=32):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(n, d)
                                   for n, d in zip(sizes, dims)])
        in_dim = sum(dims) + in_extra
        self.mlp = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(),
                                 nn.Linear(64, out_dim))
        for e in self.embs:
            nn.init.normal_(e.weight, std=0.05)

    def forward(self, idxs, extra):
        x = torch.cat([e(idxs[:, j]) for j, e in enumerate(self.embs)]
                      + [extra], dim=-1)
        return F.normalize(self.mlp(x), dim=-1)   # L2 归一化 → 点积=余弦相似度


class TwoTower(nn.Module):
    def __init__(self, n_users, n_items):
        super().__init__()
        self.user_tower = Tower([n_users, 2, 7, 21], [24, 4, 4, 8], 0)
        self.item_tower = Tower([n_items], [24], len(GENRES))
        self.genre_emb = nn.Embedding(len(GENRES) + 1, len(GENRES),
                                      padding_idx=len(GENRES))

    def user_vec(self, u_idx, u_attr):
        # 用户塔没有额外的连续特征，extra 传一个 [B, 0] 的空张量占位
        empty = torch.zeros(len(u_idx), 0, device=u_attr.device)
        return self.user_tower(u_attr[u_idx], empty)

    def item_vec(self, i_idx, gid, gmask):
        g = self.genre_emb(gid[i_idx])              # [B, 6, 18]
        g = (g * gmask[i_idx].unsqueeze(-1)).sum(1) \
            / gmask[i_idx].sum(1, keepdim=True)     # 多值类型求均值
        return self.item_tower(i_idx.unsqueeze(1), g)


# ---------------------------------------------------------------- 评估
@torch.no_grad()
def evaluate(model, train, test, u_attr, gid, gmask, maps, K=10):
    """Faiss 建库 → 全用户检索 → 排除已看 → 算 Recall@K / NDCG@K。"""
    import faiss
    model.eval()
    n_items = len(maps["i_map"])
    gid_t = torch.as_tensor(gid, device=DEVICE)
    gmask_t = torch.as_tensor(gmask, device=DEVICE)
    u_attr_t = torch.as_tensor(u_attr, device=DEVICE)

    item_vecs = model.item_vec(torch.arange(n_items, device=DEVICE),
                               gid_t, gmask_t).cpu().numpy()
    user_vecs = model.user_vec(torch.arange(len(u_attr), device=DEVICE),
                               u_attr_t).cpu().numpy()

    index = faiss.IndexFlatIP(item_vecs.shape[1])  # 内积（向量已归一化=余弦）
    index.add(item_vecs.astype(np.float32))
    _, topk = index.search(user_vecs.astype(np.float32), K + 50)  # 多取再过滤

    seen = train.groupby("u")["i"].apply(set).to_dict()
    test_by_user = test.groupby("u")["i"].first()
    hits, ndcg = 0, 0.0
    for u, true_i in test_by_user.items():
        recs = [i for i in topk[u] if i not in seen.get(u, set())][:K]
        if true_i in recs:
            hits += 1
            ndcg += 1.0 / np.log2(recs.index(true_i) + 2)
    n = len(test_by_user)
    return hits / n, ndcg / n


# ---------------------------------------------------------------- 训练
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = ExperimentConfig.from_args(__file__, args, seed=args.seed)
    cfg.save(EXP_DIR)
    print(f"实验配置已保存: {cfg.run_name}")
    train, test, u_attr, gid, gmask, maps = load_data()
    n_users, n_items = len(u_attr), len(maps["i_map"])
    print(f"设备 {DEVICE} | 用户 {n_users} 物品 {n_items} "
          f"| 正样本对：训练 {len(train)} 测试 {len(test)}")

    model = TwoTower(n_users, n_items).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    u_attr_t = torch.as_tensor(u_attr, device=DEVICE)
    gid_t = torch.as_tensor(gid, device=DEVICE)
    gmask_t = torch.as_tensor(gmask, device=DEVICE)
    pu = torch.as_tensor(train["u"].values.copy(), device=DEVICE)
    pi = torch.as_tensor(train["i"].values.copy(), device=DEVICE)
    labels = torch.arange(args.bs, device=DEVICE)

    # 采样 softmax 的 logit 校正（采样偏差校正的关键技巧）：
    # 批内候选服从训练物品分布 q(j)，而不是全库均匀分布；在 logits 中
    # 减去 log q(j)，抵消高频物品在采样候选里的过度代表。
    freq = torch.bincount(pi, minlength=n_items).float()
    log_q = torch.log(freq / freq.sum() + 1e-12)

    log_path = EXP_DIR / "two_tower_log.csv"
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, nb = time.time(), 0.0, 0
        perm = torch.randperm(len(pu), device=DEVICE)
        for s in range(0, len(perm) - args.bs, args.bs):
            idx = perm[s: s + args.bs]
            u = model.user_vec(pu[idx], u_attr_t)      # [B, 32]
            it = model.item_vec(pi[idx], gid_t, gmask_t)  # [B, 32]
            logits = u @ it.T / TEMP - log_q[pi[idx]].unsqueeze(0)  # 校正后打分
            loss = loss_fn(logits, labels)             # 对角线是正样本
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        recall, ndcg = evaluate(model, train, test, u_attr, gid, gmask, maps)
        print(f"epoch {epoch:2d} | loss {total / nb:.4f} "
              f"| Recall@10 {recall:.4f} | NDCG@10 {ndcg:.4f} "
              f"| {time.time() - t0:.1f}s")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"two_tower,{epoch},{total / nb:.6f},{recall:.6f},"
                    f"{ndcg:.6f}\n")

    torch.save({"model": model.state_dict(), "maps": maps,
                "n_users": n_users, "n_items": n_items},
               CKPT_DIR / "two_tower.pt")
    print("模型已保存，可供阶段 5 的 Demo 调用")


if __name__ == "__main__":
    main()
