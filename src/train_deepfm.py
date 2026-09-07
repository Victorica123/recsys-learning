# -*- coding: utf-8 -*-
"""
阶段 2：显式偏好预测 —— LR / FM / DeepFM 三模型对比
====================================================

【任务边界】预测已发生评分的 (用户 u, 电影 i) 是否为正向偏好
（评分≥4 → 1，否则 → 0）。MovieLens 没有曝光/未点击日志，因此这里训练的是
CTR-style 二分类排序器，不是真实点击率（CTR）模型，也不能解释为曝光后的点击概率。

【三个模型，一层比一层强】
- LR（逻辑回归）：线性加权，只认识单个特征        → 工业界的 baseline
- FM（因子分解机）：额外学习特征【两两组合】的向量点积，
  比如"18-24岁 ∩ 动作片"这种交叉模式，且参数共享不怕稀疏
- DeepFM = FM + 深度网络：FM 管低阶交叉，MLP 管高阶交叉，共享 Embedding
  （华为 2017 年提出，至今仍是很多公司的线上主力或 teacher 模型）

【评估指标】AUC：随机抽一个正样本和一个负样本，模型把正样本排前面的概率。
0.5 = 瞎猜，1.0 = 完美。它衡量已评分样本上的偏好排序，不衡量真实曝光 CTR。

【运行方式】
    .venv/Scripts/python.exe src/train_deepfm.py            # 三个模型依次训练对比
    .venv/Scripts/python.exe src/train_deepfm.py --model deepfm --epochs 10
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

EMB_DIM = 16


# ---------------------------------------------------------------- 数据
def load_data(seed=42):
    """构造显式偏好样本：每条已发生评分 → 特征 + 正向/非正向标签。

    数据不包含曝光但未评分的电影，因而不能构造真实 CTR 的负样本。
    """
    ratings = pd.read_csv(DATA / "ratings.dat", sep="::", engine="python",
                          names=["user_id", "movie_id", "rating", "timestamp"])
    users = pd.read_csv(DATA / "users.dat", sep="::", engine="python",
                        names=["user_id", "gender", "age", "occupation", "zip"])
    movies = pd.read_csv(DATA / "movies.dat", sep="::", engine="python",
                         names=["movie_id", "title", "genres"],
                         encoding="latin-1")
    df = ratings.merge(users, on="user_id").merge(movies, on="movie_id")
    df["label"] = (df["rating"] >= 4).astype(np.float32)  # ≥4星 → 喜欢(1)

    # 原始 ID → 从 0 开始的连续下标
    maps = {}
    for col in ["user_id", "movie_id", "gender", "age", "occupation"]:
        maps[col] = {v: i for i, v in enumerate(sorted(df[col].unique()))}
        df[col + "_idx"] = df[col].map(maps[col])
    g2i = {g: i for i, g in enumerate(GENRES)}
    df["genre_idx"] = df["genres"].str.split("|").apply(
        lambda gs: [g2i[g] for g in gs])

    # 按时间切分 80% 训练 / 10% 验证 / 10% 测试（最后 20% 交互做测试）。
    # 不用随机切分：同一用户的历史交互若同时出现在训练/测试集，user_id /
    # movie_id Embedding 会「背下」该用户偏好，导致 AUC 被系统性高估。
    # 时间切分让模型只能用「过去」预测「未来的已评分偏好」，减少时间穿越；
    # 但它不能消除“只观察到用户主动评分的电影”这一选择偏差。
    # 注意 maps 在切分前就基于全量数据构建，因此 field_sizes 稳定，
    # serving.py 加载时也能拿到完全一致的映射。
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train = df[: int(n * .8)].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    valid = df[int(n * .8): int(n * .9)].reset_index(drop=True)
    test = df[int(n * .9):].reset_index(drop=True)
    return train, valid, test, maps


FIELD_COLS = ["user_id_idx", "movie_id_idx", "gender_idx", "age_idx",
              "occupation_idx"]          # 5 个单值特征
FIELD_SIZES = None                        # 由 maps 推断
N_GENRE = len(GENRES)                     # 第 6 个特征是多值的类型


def to_batch(df, i0, bs, device):
    """把一批 DataFrame 行转成模型输入张量。

    多值特征（电影类型）用 padding + mask 处理：
    每行类型个数不同，统一补到本批最大长度，求均值时把 padding 遮掉。
    """
    b = df.iloc[i0: i0 + bs]
    x = torch.as_tensor(b[FIELD_COLS].values.copy(), device=device)  # [B,5]
    y = torch.as_tensor(b["label"].values.copy(), device=device)     # [B]
    lens = b["genre_idx"].str.len()
    max_len = lens.max()
    gid = torch.zeros(len(b), max_len, dtype=torch.long)
    mask = torch.zeros(len(b), max_len)
    for r, (gs, ln) in enumerate(zip(b["genre_idx"], lens)):
        gid[r, :ln] = torch.as_tensor(gs)
        mask[r, :ln] = 1.0
    return x, gid.to(device), mask.to(device), y


# ---------------------------------------------------------------- 模型
class DeepFM(nn.Module):
    """LR / FM / DeepFM 三合一：用 use_fm / use_deep 开关控制。

    所有特征共享同一套 Embedding 表（FM 的二阶交叉和 MLP 的输入
    用的是同一份向量，这正是 DeepFM 相对 Wide&Deep 的改进点）。
    """

    def __init__(self, field_sizes, dim=EMB_DIM, use_fm=True, use_deep=True):
        super().__init__()
        self.use_fm, self.use_deep = use_fm, use_deep
        # 二阶/深度用的高维 Embedding（5 个单值特征 + 1 个多值类型特征）
        self.embs = nn.ModuleList([nn.Embedding(n, dim) for n in field_sizes]
                                  + [nn.Embedding(N_GENRE + 1, dim,
                                                  padding_idx=N_GENRE)])
        # 一阶线性部分：每个特征值一个标量权重（等价于 LR 的权重）
        self.lin = nn.ModuleList([nn.Embedding(n, 1) for n in field_sizes]
                                 + [nn.Embedding(N_GENRE + 1, 1,
                                                 padding_idx=N_GENRE)])
        self.bias = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(6 * dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1))
        for e in self.embs:
            nn.init.normal_(e.weight, std=0.05)

    def field_embeddings(self, x, gid, mask, tables):
        """返回 [B, 6, D]：5 个单值特征直接查表，类型特征按 mask 求均值。"""
        vecs = [t(x[:, j]) for j, t in enumerate(tables[:-1])]
        g = tables[-1](gid)                      # [B, L, D]
        g = (g * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        return torch.stack(vecs + [g], dim=1)    # [B, 6, D]

    def forward(self, x, gid, mask):
        # 一阶：所有线性权重相加
        first = self.field_embeddings(x, gid, mask, self.lin).sum((1, 2))
        logit = self.bias + first
        emb = self.field_embeddings(x, gid, mask, self.embs)  # [B,6,D]
        if self.use_fm:
            # FM 二阶的 O(kn)  trick：0.5 * [(Σv)² - Σ(v²)]
            s, sq = emb.sum(1), (emb ** 2).sum(1)
            logit = logit + 0.5 * (s ** 2 - sq).sum(1)
        if self.use_deep:
            logit = logit + self.mlp(emb.flatten(1)).squeeze(-1)
        return logit


# ---------------------------------------------------------------- 评估
def auc_score(y_true, y_score):
    """手算 AUC（秩统计法）：正样本得分超过负样本的比例。

    并列分数用平均秩（Wilcoxon 的标准做法）。直接 argsort 会给并列值分配
    任意先后，等价于随机打破平局，��让 AUC 偏离真值。
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = y_true == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")          # 单一类别下 AUC 无定义
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # 把每段并列分数的秩替换成该段的平均秩
    sorted_scores = y_score[order]
    start = 0
    for end in range(1, len(sorted_scores) + 1):
        if end == len(sorted_scores) or sorted_scores[end] != sorted_scores[start]:
            if end - start > 1:
                ranks[order[start:end]] = (start + end + 1) / 2.0
            start = end
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def gauc_score(y_true, y_score, group_ids):
    """User-level GAUC：按用户分组算 AUC，再按每组样本数加权平均。

    为什么工业界看 GAUC 而不是全局 AUC：全局 AUC 把所有用户的样本混在一起
    排序，一个"总是给活跃用户高分、给冷启动用户低分"的模型即使在每个用户
    内部完全乱序，也能拿到很高的全局 AUC——因为跨用户的区分度掩盖了组内的
    排序质量。而线上真正决定体验的是**同一个用户的候选之间**排得对不对。

    只有一种标签的用户（全正或全负）组内 AUC 无定义，按惯例剔除，同时返回
    被剔除的比例，避免"剩下的用户样本太少"这件事被悄悄藏起来。
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    group_ids = np.asarray(group_ids)

    order = np.argsort(group_ids, kind="mergesort")
    y_true, y_score, group_ids = y_true[order], y_score[order], group_ids[order]
    # 用不等比较找分组边界，而不是 np.diff —— group_id 可能是字符串
    # （actor_id、hash 之类），字符串没有减法。
    boundaries = np.flatnonzero(group_ids[1:] != group_ids[:-1]) + 1
    weighted_sum, weight_total = 0.0, 0.0
    used_groups = skipped_groups = 0
    for chunk_true, chunk_score in zip(np.split(y_true, boundaries),
                                       np.split(y_score, boundaries)):
        group_auc = auc_score(chunk_true, chunk_score)
        if np.isnan(group_auc):
            skipped_groups += 1
            continue
        used_groups += 1
        weight = len(chunk_true)          # 曝光数加权（工业界最常见口径）
        weighted_sum += group_auc * weight
        weight_total += weight
    return {
        "gauc": weighted_sum / weight_total if weight_total else float("nan"),
        "groups_used": used_groups,
        "groups_skipped_single_class": skipped_groups,
        "coverage": (used_groups / (used_groups + skipped_groups)
                     if used_groups + skipped_groups else 0.0),
    }


@torch.no_grad()
def predict_scores(model, df, device, bs=16384):
    """返回整份数据的预测概率，供 AUC / GAUC 复用同一次前向。"""
    model.eval()
    scores = []
    for i0 in range(0, len(df), bs):
        x, gid, mask, _ = to_batch(df, i0, bs, device)
        scores.append(torch.sigmoid(model(x, gid, mask)).cpu().numpy())
    return np.concatenate(scores) if scores else np.zeros(0)


@torch.no_grad()
def evaluate_auc(model, df, device, bs=16384):
    return auc_score(df["label"].values, predict_scores(model, df, device, bs))


@torch.no_grad()
def evaluate_ranking(model, df, device, bs=16384):
    """一次前向同时给出全局 AUC 与 user-level GAUC。"""
    scores = predict_scores(model, df, device, bs)
    labels = df["label"].values
    result = {"auc": auc_score(labels, scores)}
    result.update(gauc_score(labels, scores, df["user_id_idx"].values))
    return result


# ---------------------------------------------------------------- 训练
def train_one(name, field_sizes, train, valid, test, epochs, lr, wd, bs):
    use_fm = name in ("fm", "deepfm")
    use_deep = name == "deepfm"
    model = DeepFM(field_sizes, use_fm=use_fm, use_deep=use_deep).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.BCEWithLogitsLoss()  # 正向/非正向偏好的二分类交叉熵

    log_path = EXP_DIR / "deepfm_log.csv"
    best_auc, best_state = 0.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        t0, total, nb = time.time(), 0.0, 0
        for i0 in range(0, len(train), bs):
            x, gid, mask, y = to_batch(train, i0, bs, DEVICE)
            loss = loss_fn(model(x, gid, mask), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        va = evaluate_auc(model, valid, DEVICE)
        if va > best_auc:                      # 验证集选优 = early stopping
            best_auc, best_state = va, {k: v.clone() for k, v
                                        in model.state_dict().items()}
        print(f"  [{name:6s}] epoch {epoch:2d} | loss {total / nb:.4f} "
              f"| valid AUC {va:.4f} | {time.time() - t0:.1f}s")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{name},{epoch},{total / nb:.6f},{va:.6f}\n")

    model.load_state_dict(best_state)          # 用验证集最好的权重测测试集
    test_metrics = evaluate_ranking(model, test, DEVICE)
    test_auc = test_metrics["auc"]
    print(f"  [{name:6s}] ★ 测试集 AUC = {test_auc:.4f}（验证集最佳 {best_auc:.4f}）")
    print(f"  [{name:6s}]   user-level GAUC = {test_metrics['gauc']:.4f}"
          f"（{test_metrics['groups_used']} 个用户参与，"
          f"{test_metrics['groups_skipped_single_class']} 个因单一标签剔除，"
          f"覆盖 {test_metrics['coverage']:.1%}）")
    if name == "deepfm":
        # serving 只读取 model；额外元数据把训练语义固化在 checkpoint 内，避免
        # 后续把这个分数误称为真实点击概率。
        torch.save({
            "model": best_state,
            "task": "positive_rating_prediction",
            "label": "rating >= 4 among observed MovieLens ratings",
            "split": "global_time_80_10_10",
            "is_ctr": False,
            "selection_bias": "conditional on an observed rating",
        }, CKPT_DIR / "deepfm.pt")
    return test_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all",
                    choices=["lr", "fm", "deepfm", "all"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--wd", type=float, default=1e-6)
    ap.add_argument("--bs", type=int, default=4096)
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = ExperimentConfig.from_args(__file__, args, seed=args.seed)
    cfg.save(EXP_DIR)
    print(f"实验配置已保存: {cfg.run_name}")
    train, valid, test, maps = load_data()
    field_sizes = [len(maps[c]) for c in
                   ["user_id", "movie_id", "gender", "age", "occupation"]]
    print(f"设备 {DEVICE} | 训练 {len(train)} 验证 {len(valid)} 测试 {len(test)}"
          f" | 正样本比例 {train['label'].mean():.2f}")

    todo = ["lr", "fm", "deepfm"] if args.model == "all" else [args.model]
    results = {}
    for name in todo:
        print(f"训练 {name.upper()} ...")
        results[name] = train_one(name, field_sizes, train, valid, test,
                                  args.epochs, args.lr, args.wd, args.bs)

    if len(results) > 1:
        print("\n========== 三模型对比（测试集） ==========")
        print(f"  {'模型':6s}   {'AUC':>7s}  {'GAUC':>7s}")
        for name, metrics in results.items():
            print(f"  {name.upper():6s} : {metrics['auc']:7.4f}  "
                  f"{metrics['gauc']:7.4f}")
        print("观察：FM > LR 说明特征交叉有用；DeepFM ≥ FM 说明")
        print("高阶非线性组合还能再榨出一点提升。这复用了工业 CTR 模型的架构演进，")
        print("但本实验标签来自已发生评分，不能把输出解释成真实点击率。")
        print("GAUC 明显低于 AUC 是正常的：全局 AUC 里有一部分区分度来自"
              "\"哪些用户整体更爱打高分\"，而 GAUC 只认同一用户内部的排序，"
              "更接近线上体验。")


if __name__ == "__main__":
    main()
