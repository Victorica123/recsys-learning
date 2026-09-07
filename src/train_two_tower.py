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
def split_by_time(ratings, pos, test_frac):
    """全局时间切分的纯逻辑（不读文件，便于单测）。

    返回 (train, test, cold_users)。切分点取全部交互 timestamp 的
    (1-test_frac) 分位数；测试集是每个用户在测试期的**第一个**正样本
    （"接下来他会喜欢的下一部"）。训练期完全没出现过的用户属于冷启动，
    其 embedding 从未被更新，剔除并计数。
    """
    if not 0.0 < test_frac < 1.0:
        raise ValueError("test_frac must be between 0 and 1")
    cutoff = ratings["timestamp"].quantile(1.0 - test_frac)
    train = pos[pos["timestamp"] <= cutoff]
    future = pos[pos["timestamp"] > cutoff].sort_values("timestamp")
    test = future.groupby("u").head(1)
    warm = set(train["u"].unique())
    cold_users = int((~test["u"].isin(warm)).sum())
    return train, test[test["u"].isin(warm)], cold_users


def split_leave_one_out(ratings, pos):
    """留一法的纯逻辑：每人最后一条评分做测试，只保留其中 ≥4 的。"""
    last = ratings.sort_values("timestamp").groupby("u").tail(1)
    test = last[last["rating"] >= 4]
    return pos.drop(test.index), test, 0


def load_data(split="loo", test_frac=0.1):
    """隐式反馈视角：评分≥4 记为一次'喜欢'。

    `split` 决定训练/测试怎么切，这是本项目最容易被忽视、也最容易高估指标的
    一个旋钮：

    ``loo``（默认，与阶段 1 / SASRec 复现对齐）
        每个用户按时间的最后一条评分做测试。**已知缺陷：存在时间穿越**——
        用户 A 的测试时刻可能早于用户 B 的某些训练交互，模型等于用"未来的
        全局信息"预测"过去"，会系统性高估 Recall。学术界 2020 年后对
        leave-one-out 的主要批评就是这一条。保留它是为了与既有结果可比。

    ``time``（全局时间切分，与线上一致）
        把**所有**交互按时间排序，前 (1-test_frac) 训练、后 test_frac 测试。
        模型只能用"过去"预测"未来"，没有任何时间穿越。代价是测试期里会出现
        训练期没见过的用户（冷启动），这些用户的 embedding 从未被更新过，
        必须单独剔除并报告数量，否则又变成另一种口径不清。
    """
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
    if split == "loo":
        # 与阶段 1 完全相同的评估协议：每人最新一条【评分】做测试，
        # 只保留其中 ≥4 的用户（预测"下一部喜欢的电影"）
        train, test, cold_users = split_leave_one_out(ratings, pos)
    elif split == "time":
        train, test, cold_users = split_by_time(ratings, pos, test_frac)
    else:
        raise ValueError(f"unknown split {split!r}; use 'loo' or 'time'")

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
            "i_map": {int(k): v for k, v in i_map.items()}, "g2i": g2i,
            "split": split, "cold_users_excluded": cold_users}
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

    seen = train.groupby("u")["i"].apply(set).to_dict()
    # 检索缓冲区必须够大：过滤掉已看过的物品后仍要凑满 K 条。固定多取 50 个
    # 时，重度用户（最多 1434 个正样本）会被截断到不足 K 条，Recall 被系统性
    # 低估——实测 3552 个测试用户里有 243 个拿不满 10 条。这里按最大历史长度
    # 取缓冲，口径与 `src/serving.py` 的 `n_candidates + len(seen)` 对齐，
    # 避免"离线评估比线上少召回"这类评估-服务不一致。
    max_seen = max((len(items) for items in seen.values()), default=0)
    search_k = min(K + max_seen, n_items)
    _, topk = index.search(user_vecs.astype(np.float32), search_k)

    test_by_user = test.groupby("u")["i"].first()
    hits, ndcg, truncated = 0, 0.0, 0
    for u, true_i in test_by_user.items():
        recs = [i for i in topk[u] if i >= 0 and i not in seen.get(u, set())][:K]
        if len(recs) < K:
            truncated += 1
        if true_i in recs:
            hits += 1
            ndcg += 1.0 / np.log2(recs.index(true_i) + 2)
    n = len(test_by_user)
    if truncated:
        print(f"    warn: {truncated}/{n} 个用户候选不足 {K} 条"
              f"（search_k={search_k}），Recall 会被低估")
    return hits / n, ndcg / n


# ---------------------------------------------------------------- 训练
def fit_two_tower(train_pairs, u_attr, gid, gmask, n_users, n_items, *,
                  epochs=20, lr=0.005, bs=1024, eval_fn=None,
                  log_path=None, verbose=True):
    """训练一个双塔并返回模型。抽成函数是为了让 K 折交叉拟合能复用它。

    `eval_fn(model) -> (recall, ndcg)` 可选：给了就每个 epoch 评估一次。
    交叉拟合训练的是"只用来算特征"的辅助模型，不需要逐 epoch 评估，
    传 None 能省掉每轮的 Faiss 建库+全用户检索，快很多。
    """
    model = TwoTower(n_users, n_items).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    u_attr_t = torch.as_tensor(u_attr, device=DEVICE)
    gid_t = torch.as_tensor(gid, device=DEVICE)
    gmask_t = torch.as_tensor(gmask, device=DEVICE)
    pu = torch.as_tensor(train_pairs["u"].values.copy(), device=DEVICE)
    pi = torch.as_tensor(train_pairs["i"].values.copy(), device=DEVICE)
    labels = torch.arange(bs, device=DEVICE)

    # 采样 softmax 的 logit 校正（采样偏差校正的关键技巧）：
    # 批内候选服从训练物品分布 q(j)，而不是全库均匀分布；在 logits 中
    # 减去 log q(j)，抵消高频物品在采样候选里的过度代表。
    freq = torch.bincount(pi, minlength=n_items).float()
    log_q = torch.log(freq / freq.sum() + 1e-12)

    for epoch in range(1, epochs + 1):
        model.train()
        t0, total, nb = time.time(), 0.0, 0
        perm = torch.randperm(len(pu), device=DEVICE)
        for s in range(0, len(perm) - bs, bs):
            idx = perm[s: s + bs]
            u = model.user_vec(pu[idx], u_attr_t)          # [B, 32]
            it = model.item_vec(pi[idx], gid_t, gmask_t)   # [B, 32]
            logits = u @ it.T / TEMP - log_q[pi[idx]].unsqueeze(0)  # 校正后打分
            loss = loss_fn(logits, labels)                 # 对角线是正样本
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        if eval_fn is None:
            if verbose:
                print(f"epoch {epoch:2d} | loss {total / nb:.4f} "
                      f"| {time.time() - t0:.1f}s", flush=True)
            continue
        recall, ndcg = eval_fn(model)
        if verbose:
            print(f"epoch {epoch:2d} | loss {total / nb:.4f} "
                  f"| Recall@10 {recall:.4f} | NDCG@10 {ndcg:.4f} "
                  f"| {time.time() - t0:.1f}s", flush=True)
        if log_path is not None:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"two_tower,{epoch},{total / nb:.6f},{recall:.6f},"
                        f"{ndcg:.6f}\n")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    ap.add_argument("--split", choices=["loo", "time"], default="loo",
                    help="loo=每人最后一条做测试（与既有结果可比，但有时间"
                         "穿越）；time=全局时间切分（无穿越，更接近线上）")
    ap.add_argument("--test-frac", type=float, default=0.1,
                    help="--split time 时测试期占全部交互的比例")
    ap.add_argument("--ckpt", default=None,
                    help="权重输出路径；默认 checkpoints/two_tower.pt。"
                         "跑非默认协议时务必换名，别覆盖线上产物")
    ap.add_argument("--no-save", action="store_true",
                    help="只评估不落盘（对照实验用）")
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = ExperimentConfig.from_args(__file__, args, seed=args.seed)
    cfg.save(EXP_DIR)
    print(f"实验配置已保存: {cfg.run_name}")
    train, test, u_attr, gid, gmask, maps = load_data(
        split=args.split, test_frac=args.test_frac)
    n_users, n_items = len(u_attr), len(maps["i_map"])
    print(f"设备 {DEVICE} | 用户 {n_users} 物品 {n_items} "
          f"| 切分 {args.split} | 正样本对：训练 {len(train)} 测试 {len(test)}")
    if maps["cold_users_excluded"]:
        print(f"  注意：{maps['cold_users_excluded']} 个用户只出现在测试期"
              f"（冷启动，embedding 未训练），已从评估中剔除")

    log_path = EXP_DIR / "two_tower_log.csv" if args.split == "loo" else None
    model = fit_two_tower(
        train, u_attr, gid, gmask, n_users, n_items,
        epochs=args.epochs, lr=args.lr, bs=args.bs,
        eval_fn=lambda m: evaluate(m, train, test, u_attr, gid, gmask, maps),
        log_path=log_path)

    if args.no_save:
        print("--no-save：不落盘")
        return
    ckpt_path = Path(args.ckpt) if args.ckpt else CKPT_DIR / "two_tower.pt"
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    torch.save({"model": model.state_dict(), "maps": maps,
                "n_users": n_users, "n_items": n_items,
                "split": args.split},
               ckpt_path)
    print(f"模型已保存到 {ckpt_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
