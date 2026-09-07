# -*- coding: utf-8 -*-
"""重训精排层：让训练分布对齐服务分布（修复端到端负增量）。

【要解决的问题】
`scripts/eval_end_to_end.py` 量出：线上链路（双塔召回 50 → DeepFM 精排 → Top-10）
的 Recall@10 = 0.065，**比直接取双塔前 10 名的 0.098 还差 33.6%**。诊断显示精排
把候选坍缩到热门头部（Top-10 去重物品 1853 → 773）。

根因是 **sample selection bias**：原版 DeepFM 的训练样本是全库分布的
(user, item) 对——里面既有好电影也有烂电影，学会区分"电影整体质量"就能拿到
AUC 0.754。但线上它只对双塔选出的 50 个高相关候选打分，此时候选**全是好电影**，
需要的是"对这个用户而言谁更相关"。这个分布它从没训练过。

【修法：难负样本对齐】
负样本不再从全库随机取，而是**从双塔真实会召回的候选里取**——即模型在线上真正
要面对的那批对手。这是工业界"hard negative mining"的标准做法。

【三个版本（每版只动一件事，便于归因）】
  v1 `--loss bce`      架构/特征完全不变，只把负样本换成召回候选
  v2 `--loss listwise` 在 v1 基础上换成候选集内 softmax（学排序而非绝对概率）
  v3 `--cross`         在最优损失上加用户-物品交叉特征（打破 item bias 主导）

v1/v2 的权重与原版 `DeepFM` 完全兼容，可直接用
`eval_end_to_end.py --deepfm-ckpt` 替换评估，召回与协议一字不变。
v3 用 `CrossDeepFM`（多两维稠密特征），由 checkpoint 里的 `arch` 字段区分。

【训练/验证切分（严禁碰测试集）】
测试集 = 每个用户按时间的最后一条评分（≥4），与 `train_two_tower.py` 一致。
本脚本再从**剩下的**正样本里，把每个用户时间上最后一条留作验证正样本，
其余用于训练——协议与测试同构（都是"预测下一个"），但不重叠。

【运行】
    .venv/Scripts/python.exe src/train_reranker.py --tag v1 --loss bce
    .venv/Scripts/python.exe src/train_reranker.py --tag v2 --loss listwise
    .venv/Scripts/python.exe src/train_reranker.py --tag v3 --loss listwise --cross
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from checkpoint_io import load_torch_checkpoint
from config import set_seed
from constants import GENRES, GENRE_TO_IDX
from train_deepfm import DeepFM, load_data as dfm_load
from train_two_tower import (TwoTower, fit_two_tower,
                             load_data as tt_load)

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "checkpoints"
EXP_DIR = ROOT / "experiments"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 每个正样本配多少个来自召回候选的难负样本。
NEG_PER_POS = 8
# 从双塔取多深的候选作为负样本池。线上只用 50，这里取更深是为了让负样本
# 覆盖"排序边界附近"的物品，而不是只有最容易的那 50 个。
NEG_POOL_DEPTH = 200


class CrossDeepFM(nn.Module):
    """DeepFM + 两维显式用户-物品交叉稠密特征（v3）。

    原版的 6 个字段里没有任何一维直接刻画"这个用户和这个物品有多配"——
    user_id 和 movie_id 各自是独立的 embedding，交叉完全依赖 FM 二阶项和
    MLP 自己学出来。在 ML-1M 这种小数据上，模型更容易退化成学 item bias
    （"这部电影整体好不好"），这正是端到端评估里观察到的热门坍缩。

    这里把两个信号显式喂进去：
      - `tt_score`   双塔的用户-物品余弦相似度（召回阶段的相关性打分）
      - `genre_affinity` 用户历史喜欢的类型分布 与 物品类型的点积

    `residual=True`（v5）改成显式残差形式：

        score = w_retrieval * tt_score + gate * deepfm_logit

    `gate` 初始化为 0、`w_retrieval` 初始化为 1，因此**训练开始时输出与召回
    顺序完全一致**。精排从"复刻召回"起步，只能在此之上做增量，最坏情况退化
    成召回本身而不是把它推翻。v1~v4 每一版都比"不精排"更差，根因就是它们
    从一个与召回无关的随机初始化出发，得自己重新学出召回已经编码好的相关性。
    """

    def __init__(self, field_sizes, dim=16, n_dense=2, residual=False):
        super().__init__()
        self.base = DeepFM(field_sizes, dim=dim, use_fm=True, use_deep=True)
        self.n_dense = int(n_dense)
        self.residual = bool(residual)
        self.dense = nn.Sequential(
            nn.Linear(self.n_dense, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1))
        # 零初始化最后一层：训练开始时 v3 与 v2 完全等价，交叉特征的贡献
        # 从 0 开始长出来，避免"新特征刚接上就把已有信号冲乱"。
        nn.init.zeros_(self.dense[-1].weight)
        nn.init.zeros_(self.dense[-1].bias)
        if self.residual:
            # 召回相似度是余弦（约 ±1），除以温度放大成可用的 logit 尺度。
            self.retrieval_weight = nn.Parameter(torch.tensor(20.0))
            self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x, gid, mask, dense=None):
        learned = self.base(x, gid, mask)
        if dense is not None:
            learned = learned + self.dense(dense).squeeze(-1)
        if not self.residual:
            return learned
        if dense is None:
            raise ValueError("residual ranker requires the dense features")
        return self.retrieval_weight * dense[:, 0] + self.gate * learned


# ------------------------------------------------------------------ 数据
def _retrieve(model, u_attr, gid, gmask, n_items, depth):
    """用给定双塔为所有用户取 top-depth，返回 (pool 下标, 相似度)。"""
    import faiss
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        item_vecs = model.item_vec(
            torch.arange(n_items, device=device),
            torch.as_tensor(gid, device=device),
            torch.as_tensor(gmask, device=device)).cpu().numpy().astype(np.float32)
        user_vecs = model.user_vec(
            torch.arange(len(u_attr), device=device),
            torch.as_tensor(u_attr, device=device)).cpu().numpy().astype(np.float32)
    index = faiss.IndexFlatIP(item_vecs.shape[1])
    index.add(item_vecs)
    scores, pool = index.search(user_vecs, min(depth, n_items))
    return pool, scores


def build_candidate_pools(tt_train, maps, depth=NEG_POOL_DEPTH):
    """用**线上那个**双塔为每个用户取 top-`depth` 候选，作为难负样本池。

    返回 (pool[n_users, depth] 的物品下标, tt_score[n_users, depth] 相似度)。
    """
    ckpt = load_torch_checkpoint(CKPT_DIR / "two_tower.pt")
    model = TwoTower(ckpt["n_users"], ckpt["n_items"])
    model.load_state_dict(ckpt["model"])
    _train, _test, u_attr, gid, gmask, _maps = tt_load()
    return _retrieve(model, u_attr, gid, gmask, ckpt["n_items"], depth)


def build_oof_pools(tt_train, u_attr, gid, gmask, n_users, n_items, *,
                    folds, depth, epochs, seed):
    """K 折交叉拟合：用**没见过该样本**的双塔来算它的候选池和 tt_score。

    这修的是叠加模型（stacking）的经典陷阱。线上那个双塔是用**全部**训练
    交互训好的，所以它对训练正样本的打分是**乐观偏置**的——模型当初就是被
    优化成"让 u·i 尽量高"的。精排把 tt_score 当特征学时，学到的权重是在这个
    乐观分布上拟合的；可测试集里的目标物品双塔从没训过，特征分布变了，
    权重就偏了。

    交叉拟合的做法：把训练交互切成 K 折，第 k 个双塔只用其它 K-1 折训练，
    然后**只用它来给第 k 折的样本算特征**。这样每个训练样本的 tt_score 都
    来自一个没见过它的模型——与线上"目标物品是未见过的"完全同构。

    返回 (fold_of_index, pools)：
      fold_of_index  {tt_train 的行 index: 折号}
      pools          {折号: (pool 数组, score 数组)}
    """
    rng = np.random.default_rng(seed)
    fold_assignment = rng.integers(0, folds, size=len(tt_train))
    fold_of_index = dict(zip(tt_train.index.tolist(),
                             fold_assignment.tolist()))
    pools = {}
    for k in range(folds):
        subset = tt_train[fold_assignment != k]
        print(f"  fold {k + 1}/{folds}: training two-tower on "
              f"{len(subset)} pairs (holding out {len(tt_train) - len(subset)})",
              flush=True)
        model = fit_two_tower(
            subset, u_attr, gid, gmask, n_users, n_items,
            epochs=epochs, eval_fn=None, verbose=False)
        pools[k] = _retrieve(model, u_attr, gid, gmask, n_items, depth)
    return fold_of_index, pools


def split_train_valid(tt_train):
    """每个用户时间上最后一条正样本留作验证，其余用于训练。

    测试集（每人最最后一条评分）此前已被 `train_two_tower.load_data` 剔除，
    所以这里拿到的"最后一条"是**倒数第二条**，与测试集不重叠。
    """
    ordered = tt_train.sort_values("timestamp")
    last_index = ordered.groupby("u").tail(1).index
    valid = tt_train.loc[last_index]
    train = tt_train.drop(last_index)
    return train, valid


def user_genre_affinity(train_pairs, item_genre_rows, n_users):
    """用户历史喜欢的类型分布（按行归一化），用于 v3 的交叉特征。"""
    affinity = np.zeros((n_users, len(GENRES)), dtype=np.float32)
    for u, i in zip(train_pairs["u"].values, train_pairs["i"].values):
        affinity[u] += item_genre_rows[i]
    totals = affinity.sum(axis=1, keepdims=True)
    return affinity / np.maximum(totals, 1e-6)


def item_genre_matrix(maps):
    """物品 × 类型 的 multi-hot（行内归一化），下标与双塔的 i 对齐。"""
    import pandas as pd
    movies = pd.read_csv(ROOT / "data" / "ml-1m" / "movies.dat", sep="::",
                         engine="python",
                         names=["movie_id", "title", "genres"],
                         encoding="latin-1").set_index("movie_id")
    rows = np.zeros((len(maps["i_map"]), len(GENRES)), dtype=np.float32)
    for movie_id, i in maps["i_map"].items():
        if movie_id in movies.index:
            for genre in movies.loc[movie_id, "genres"].split("|"):
                rows[i, GENRE_TO_IDX.get(genre, 0)] = 1.0
    totals = rows.sum(axis=1, keepdims=True)
    return rows / np.maximum(totals, 1e-6)


def sample_groups(pairs, pool, positives_by_user, rng, neg_per_pos,
                  serve_k=None):
    """为每个正样本采 `neg_per_pos` 个来自召回候选、且用户没喜欢过的负样本。

    返回 (groups[n, 1+neg] 的物品下标, users[n]) —— 每行第 0 列是正样本。
    这种"一正多负"的分组同时支撑 pointwise BCE 与 listwise softmax，
    两个版本因此吃的是**完全相同的样本**，差异只在损失函数。

    `serve_k` 是关键的第二层对齐。线上 `Recommender.recommend` 的候选是
    "双塔全库检索 → 剔除已看 → **取前 serve_k 个**"，即精排面对的是**最难的
    那 serve_k 个对手**。若负样本从 top-200 里均匀采，训练就比服务简单，
    模型没在真正的决策边界上被训练过——这与原版"全库随机负样本"是同一类
    错误，只是浅了一层。给了 `serve_k` 就复刻这个截断：先按线上口径剔除该
    用户的其它正样本（目标物品在线上是未看过的，所以保留），再截前
    `serve_k` 个。
    """
    groups, users = [], []
    for u, i in zip(pairs["u"].values, pairs["i"].values):
        liked = positives_by_user.get(u)
        if liked is None:
            continue
        candidates = pool[u]
        if serve_k:
            # 线上会滤掉用户已看过的物品；目标物品此刻扮演"未看过的下一个"，
            # 所以只滤掉其它正样本，再取前 serve_k 个——与服务逐字一致。
            others = liked - {int(i)}
            served = candidates[~np.isin(candidates, list(others))][:serve_k]
            negatives = served[served != i]
        else:
            negatives = candidates[~np.isin(candidates, list(liked))]
        if len(negatives) < neg_per_pos:
            continue
        chosen = rng.choice(negatives, size=neg_per_pos, replace=False)
        groups.append(np.concatenate([[i], chosen]))
        users.append(u)
    return np.asarray(groups, dtype=np.int64), np.asarray(users, dtype=np.int64)


def build_feature_tensors(groups, users, *, maps, dfm_maps, users_df,
                          idx2movie, movie_genre_idx):
    """把 (user, item) 分组展开成 DeepFM 的字段张量。"""
    flat_items = groups.reshape(-1)
    flat_users = np.repeat(users, groups.shape[1])
    idx2user = {v: k for k, v in maps["u_map"].items()}

    user_fields = {}
    for u in np.unique(flat_users):
        user_id = idx2user[int(u)]
        info = users_df.loc[user_id]
        user_fields[int(u)] = (
            dfm_maps["user_id"][user_id],
            dfm_maps["gender"][info.gender],
            dfm_maps["age"][info.age],
            dfm_maps["occupation"][info.occupation])

    x = np.empty((len(flat_items), 5), dtype=np.int64)
    for row, (u, i) in enumerate(zip(flat_users, flat_items)):
        uid_idx, gender_idx, age_idx, occ_idx = user_fields[int(u)]
        movie_id = idx2movie[int(i)]
        x[row] = (uid_idx, dfm_maps["movie_id"][movie_id],
                  gender_idx, age_idx, occ_idx)

    max_len = max(len(movie_genre_idx[int(i)]) for i in flat_items)
    gid = np.zeros((len(flat_items), max_len), dtype=np.int64)
    mask = np.zeros((len(flat_items), max_len), dtype=np.float32)
    for row, i in enumerate(flat_items):
        genres = movie_genre_idx[int(i)]
        gid[row, :len(genres)] = genres
        mask[row, :len(genres)] = 1.0
    return (torch.from_numpy(x), torch.from_numpy(gid),
            torch.from_numpy(mask), flat_users, flat_items)


def build_dense_features(flat_users, flat_items, tt_scores, affinity,
                         item_genres):
    """v3 的两维交叉特征：双塔相似度 + 用户类型偏好与物品类型的匹配度。"""
    similarity = np.einsum(
        "ij,ij->i",
        affinity[flat_users], item_genres[flat_items]).astype(np.float32)
    return torch.from_numpy(np.stack([
        tt_scores.astype(np.float32), similarity], axis=1))


# ------------------------------------------------------------------ 训练
def group_scores(model, tensors, dense, group_width, bs, train=False):
    """按分组返回 [n_groups, group_width] 的打分矩阵。"""
    x, gid, mask = tensors
    outputs = []
    for start in range(0, len(x), bs):
        chunk = slice(start, start + bs)
        batch_dense = None if dense is None else dense[chunk].to(DEVICE)
        logits = model(x[chunk].to(DEVICE), gid[chunk].to(DEVICE),
                       mask[chunk].to(DEVICE), *(
                           () if batch_dense is None else (batch_dense,)))
        outputs.append(logits if train else logits.detach())
    return torch.cat(outputs).view(-1, group_width)


@torch.no_grad()
def validate(model, tensors, dense, group_width, bs, k=10):
    """验证指标 = 正样本在"1 正 + N 难负"里排进前 k 的比例（含 MRR）。

    **验证的分组宽度必须等于线上的候选池大小**（默认 50），否则 Hit@10 会
    退化成常数 1：8 个负样本时"排进前 10"是必然事件，那样选出来的 epoch
    和最终目标毫无关系。训练用少量负样本是为了效率，验证必须还原服务条件。

    这与端到端评估同构：都是"在 50 个召回候选里把真实物品排进 Top-10"。
    用它选 epoch，比用全局 AUC 选更接近最终目标——原版正是被全局 AUC 误导的。
    """
    model.eval()
    scores = group_scores(model, tensors, dense, group_width, bs)
    # 第 0 列是正样本，统计有多少个负样本分数比它高
    better = (scores[:, 1:] > scores[:, :1]).sum(dim=1)
    hit_rate = (better < k).float().mean().item()
    mrr = (1.0 / (better.float() + 1.0)).mean().item()
    return hit_rate, mrr


def train_model(model, train_tensors, train_dense, valid_tensors, valid_dense,
                *, group_width, valid_width, loss_kind, epochs, lr, wd, bs,
                log_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    n_groups = len(train_tensors[0]) // group_width
    labels = torch.zeros(0, dtype=torch.long)
    best = {"hit_rate": -1.0, "state": None, "epoch": 0, "mrr": 0.0}
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(n_groups)          # 每个 epoch 重新打散分组
        total_loss, n_batches = 0.0, 0
        started = time.perf_counter()
        group_bs = max(1, bs // group_width)
        for start in range(0, n_groups, group_bs):
            picked = order[start: start + group_bs]
            rows = (picked.unsqueeze(1) * group_width
                    + torch.arange(group_width)).reshape(-1)
            x, gid, mask = (tensor[rows] for tensor in train_tensors)
            dense = None if train_dense is None else train_dense[rows].to(DEVICE)
            logits = model(x.to(DEVICE), gid.to(DEVICE), mask.to(DEVICE),
                           *(() if dense is None else (dense,)))
            grouped = logits.view(-1, group_width)
            if loss_kind == "listwise":
                # 候选集内 softmax：只要求"正样本得分最高"，不要求它逼近 1。
                # 这正是精排在线上被问的问题。
                if len(labels) != len(grouped):
                    labels = torch.zeros(len(grouped), dtype=torch.long,
                                         device=DEVICE)
                loss = F.cross_entropy(grouped, labels[:len(grouped)])
            else:
                target = torch.zeros_like(grouped)
                target[:, 0] = 1.0
                loss = F.binary_cross_entropy_with_logits(grouped, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        hit_rate, mrr = validate(model, valid_tensors, valid_dense,
                                 valid_width, bs)
        row = {"epoch": epoch, "loss": round(total_loss / n_batches, 6),
               "valid_hit@10": round(hit_rate, 6), "valid_mrr": round(mrr, 6),
               "seconds": round(time.perf_counter() - started, 1)}
        history.append(row)
        print(f"  epoch {epoch:2d} | loss {row['loss']:.4f} "
              f"| valid Hit@10 {hit_rate:.4f} | MRR {mrr:.4f} "
              f"| {row['seconds']:.1f}s", flush=True)
        if hit_rate > best["hit_rate"]:
            best = {"hit_rate": hit_rate, "mrr": mrr, "epoch": epoch,
                    "state": {k: v.detach().clone()
                              for k, v in model.state_dict().items()}}

    with log_path.open("x", newline="", encoding="utf-8") as handle:
        handle.write("epoch,loss,valid_hit@10,valid_mrr,seconds\n")
        for row in history:
            handle.write(",".join(str(row[key]) for key in
                                  ("epoch", "loss", "valid_hit@10",
                                   "valid_mrr", "seconds")) + "\n")
    return best, history


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True, help="产物标签；已存在则拒绝覆盖")
    parser.add_argument("--loss", choices=["bce", "listwise"], default="bce")
    parser.add_argument("--cross", action="store_true",
                        help="加入用户-物品交叉稠密特征（v3）")
    parser.add_argument("--residual", action="store_true",
                        help="残差形式：从复刻召回顺序起步，只学增量（v5）；"
                             "隐含 --cross，因为要用召回分数当基底")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-6)
    parser.add_argument("--bs", type=int, default=4096)
    parser.add_argument("--neg", type=int, default=NEG_PER_POS)
    parser.add_argument("--valid-neg", type=int, default=49,
                        help="验证分组的负样本数；默认 49，使分组宽度 50 "
                             "与线上候选池一致，Hit@10 才有意义")
    parser.add_argument("--pool-depth", type=int, default=NEG_POOL_DEPTH)
    parser.add_argument("--oof-folds", type=int, default=0,
                        help="K 折交叉拟合特征（v6）：>1 时训练 K 个双塔，"
                             "每个样本的 tt_score 都由没见过它的模型算出，"
                             "消除叠加模型的乐观偏置。0/1 = 关闭")
    parser.add_argument("--oof-epochs", type=int, default=20,
                        help="每个交叉拟合双塔的训练轮数")
    parser.add_argument("--serve-k", type=int, default=0,
                        help="按线上候选池截断负样本（线上是 50）；"
                             "0 = 从整个 pool 均匀采（v1~v3 的做法）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pos-per-user", type=int, default=60,
                        help="每用户参与训练的正样本上限，避免重度用户主导")
    args = parser.parse_args()

    checkpoint_path = CKPT_DIR / f"deepfm_rerank_{args.tag}.pt"
    log_path = EXP_DIR / f"rerank_{args.tag}_train.csv"
    report_path = EXP_DIR / f"rerank_{args.tag}_report.json"
    existing = [str(p) for p in (checkpoint_path, log_path, report_path)
                if p.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite; choose a new --tag:\n" + "\n".join(existing))

    if args.residual and not args.cross:
        args.cross = True          # 残差基底就是召回分数，必须有稠密特征
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    print(f"Device {DEVICE} | loss={args.loss} | cross={args.cross}", flush=True)

    import pandas as pd
    tt_train, _tt_test, u_attr, gid, gmask, maps = tt_load()
    _dfm_train, _dfm_valid, _dfm_test, dfm_maps = dfm_load()
    users_df = pd.read_csv(ROOT / "data" / "ml-1m" / "users.dat", sep="::",
                           engine="python",
                           names=["user_id", "gender", "age", "occupation",
                                  "zip"]).set_index("user_id")
    idx2movie = {v: k for k, v in maps["i_map"].items()}

    n_users, n_items = len(u_attr), len(maps["i_map"])
    fold_of_index, fold_pools = None, None
    if args.oof_folds > 1:
        print(f"Cross-fitting {args.oof_folds} two-towers for out-of-fold "
              f"features (depth={args.pool_depth}) ...", flush=True)
        fold_of_index, fold_pools = build_oof_pools(
            tt_train, u_attr, gid, gmask, n_users, n_items,
            folds=args.oof_folds, depth=args.pool_depth,
            epochs=args.oof_epochs, seed=args.seed)
        # 线上那个双塔仍然负责"服务时"的候选与打分；交叉拟合只改**训练特征**。
        pool, pool_scores = build_candidate_pools(
            tt_train, maps, args.pool_depth)
    else:
        print("Retrieving two-tower candidate pools "
              f"(depth={args.pool_depth}) ...", flush=True)
        pool, pool_scores = build_candidate_pools(
            tt_train, maps, args.pool_depth)

    def make_score_lookup(pool_array, score_array):
        return [dict(zip(pool_array[u].tolist(), score_array[u].tolist()))
                for u in range(pool_array.shape[0])]

    score_lookup = make_score_lookup(pool, pool_scores)
    fold_score_lookup = (
        {k: make_score_lookup(*fold_pools[k]) for k in fold_pools}
        if fold_pools else None)

    train_pairs, valid_pairs = split_train_valid(tt_train)
    if args.max_pos_per_user:
        train_pairs = (train_pairs.sort_values("timestamp")
                       .groupby("u").tail(args.max_pos_per_user))
    positives_by_user = tt_train.groupby("u")["i"].apply(set).to_dict()
    print(f"Train positives {len(train_pairs)} | valid positives "
          f"{len(valid_pairs)} | {args.neg} hard negatives each", flush=True)

    movie_genre_idx = {}
    movies = pd.read_csv(ROOT / "data" / "ml-1m" / "movies.dat", sep="::",
                         engine="python",
                         names=["movie_id", "title", "genres"],
                         encoding="latin-1").set_index("movie_id")
    for movie_id, i in maps["i_map"].items():
        genres = ([GENRE_TO_IDX.get(g, 0)
                   for g in movies.loc[movie_id, "genres"].split("|")]
                  if movie_id in movies.index else [0])
        movie_genre_idx[i] = genres

    group_width = args.neg + 1
    valid_width = args.valid_neg + 1
    packs = {}
    for name, pairs, n_neg in (("train", train_pairs, args.neg),
                               ("valid", valid_pairs, args.valid_neg)):
        if fold_pools is None:
            groups, group_users = sample_groups(
                pairs, pool, positives_by_user, rng, n_neg,
                serve_k=args.serve_k or None)
            group_folds = None
        else:
            # 按折分批采样：每个样本的候选池来自没见过它的那个双塔。
            group_chunks, user_chunks, fold_chunks = [], [], []
            pair_folds = pairs.index.map(fold_of_index)
            for k in range(args.oof_folds):
                fold_pairs = pairs[pair_folds == k]
                if fold_pairs.empty:
                    continue
                g, gu = sample_groups(
                    fold_pairs, fold_pools[k][0], positives_by_user, rng,
                    n_neg, serve_k=args.serve_k or None)
                if len(g) == 0:
                    continue
                group_chunks.append(g)
                user_chunks.append(gu)
                fold_chunks.append(np.full(len(g), k, dtype=np.int64))
            groups = np.concatenate(group_chunks)
            group_users = np.concatenate(user_chunks)
            group_folds = np.concatenate(fold_chunks)
        tensors = build_feature_tensors(
            groups, group_users, maps=maps, dfm_maps=dfm_maps,
            users_df=users_df, idx2movie=idx2movie,
            movie_genre_idx=movie_genre_idx)
        packs[name] = (tensors[:3], tensors[3], tensors[4], len(groups),
                       group_folds)
        print(f"  {name}: {len(groups)} groups x {n_neg + 1}"
              + ("" if group_folds is None else " (out-of-fold features)"),
              flush=True)

    dense_packs = {"train": None, "valid": None}
    if args.cross:
        item_genres = item_genre_matrix(maps)
        affinity = user_genre_affinity(tt_train, item_genres, n_users)
        for name in ("train", "valid"):
            _tensors, flat_users, flat_items, _n, group_folds = packs[name]
            if group_folds is None:
                tt_scores = np.array(
                    [score_lookup[int(u)].get(int(i), 0.0)
                     for u, i in zip(flat_users, flat_items)], dtype=np.float32)
            else:
                # 每个分组用自己那一折的打分表；同一组内的候选打分因此同源、
                # 可比，而且都来自一个没见过该正样本的模型。
                width = group_width if name == "train" else valid_width
                flat_folds = np.repeat(group_folds, width)
                tt_scores = np.array(
                    [fold_score_lookup[int(f)][int(u)].get(int(i), 0.0)
                     for f, u, i in zip(flat_folds, flat_users, flat_items)],
                    dtype=np.float32)
            dense_packs[name] = build_dense_features(
                flat_users, flat_items, tt_scores, affinity, item_genres)

    field_sizes = [len(dfm_maps[c]) for c in
                   ["user_id", "movie_id", "gender", "age", "occupation"]]
    model = (CrossDeepFM(field_sizes, residual=args.residual) if args.cross
             else DeepFM(field_sizes)).to(DEVICE)
    if not args.cross:
        # 让 DeepFM 与 CrossDeepFM 有同样的调用签名，训练循环不必分支
        base_forward = model.forward

        def forward(x, gid, mask, dense=None):
            return base_forward(x, gid, mask)

        model.forward = forward

    print(f"Training {'CrossDeepFM' if args.cross else 'DeepFM'} "
          f"({args.epochs} epochs) ...", flush=True)
    best, history = train_model(
        model, packs["train"][0], dense_packs["train"],
        packs["valid"][0], dense_packs["valid"],
        group_width=group_width, valid_width=valid_width,
        loss_kind=args.loss, epochs=args.epochs,
        lr=args.lr, wd=args.wd, bs=args.bs, log_path=log_path)

    state = best["state"]
    if args.cross:
        # CrossDeepFM 的 DeepFM 子模块前缀是 base.*，剥掉后与原版权重同构，
        # 便于需要时回退成纯 DeepFM 加载。
        payload = {"model": state, "arch": "cross_deepfm",
                   "residual": bool(args.residual),
                   "base_state": {k[len("base."):]: v
                                  for k, v in state.items()
                                  if k.startswith("base.")}}
    else:
        payload = {"model": state, "arch": "deepfm"}
    payload["config"] = vars(args)
    torch.save(payload, checkpoint_path)

    report = {
        "experiment": "retrieval-aligned reranker training",
        "motivation": ("original DeepFM was trained on full-catalog random "
                       "pairs but serves only two-tower candidates; "
                       "end-to-end Recall@10 was 0.065 vs 0.098 without it"),
        "config": vars(args),
        "group_width": group_width,
        "valid_group_width": valid_width,
        "valid_metric_note": (
            "valid Hit@10 ranks the held-out positive against "
            f"{args.valid_neg} hard negatives, i.e. the same 50-candidate "
            "condition the ranker faces at serving time"),
        "train_groups": packs["train"][3],
        "valid_groups": packs["valid"][3],
        "oof_folds": args.oof_folds,
        "oof_note": (
            "tt_score for every training/validation sample came from a "
            "two-tower that never trained on that interaction; the serving "
            "path still uses the full two-tower, which likewise has never "
            "seen the test item - so train and serve are now aligned"
            if args.oof_folds > 1 else
            "tt_score came from the full two-tower, which trained on the "
            "positives it scores (optimistically biased)"),
        "best_epoch": best["epoch"],
        "best_valid_hit@10": round(best["hit_rate"], 6),
        "best_valid_mrr": round(best["mrr"], 6),
        "history": history,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "next_step": (f"eval_end_to_end.py --tag <new> --deepfm-ckpt "
                      f"{checkpoint_path.relative_to(ROOT)}"),
    }
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"\nBest epoch {best['epoch']} | valid Hit@10 "
          f"{best['hit_rate']:.4f} | MRR {best['mrr']:.4f}")
    print(f"Checkpoint: {checkpoint_path.relative_to(ROOT)}")
    print(f"Next: .venv/Scripts/python.exe scripts/eval_end_to_end.py "
          f"--tag <new-tag> --deepfm-ckpt "
          f"{checkpoint_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
