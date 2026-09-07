# -*- coding: utf-8 -*-
"""
Phase 2 · 模块 1：SASRec 论文复现
==================================

复现论文：Kang & McAuley, *Self-Attentive Sequential Recommendation*,
ICDM 2018（简称 SASRec —— 用单向 Transformer 做序列推荐的经典之作）

【和 Phase 1 的本质区别】
DeepFM/双塔把每个 (用户, 电影) 当独立样本；SASRec 把用户的历史行为
【序列】喂给 Transformer，让注意力自己学"刚才看了什么对下一步最重要"。

【论文关键设置（严格复现）】
- 隐式反馈：所有评分都视为一次交互（不区分高低分）
- 数据划分：每个用户最后 1 个交互做测试、倒数第 2 个做验证（leave-one-out）
- 评估：目标物品 + 100 个随机负样本一起排序，算 HR@10 / NDCG@10
- 模型：物品 Embedding + 可学习位置 Embedding → N 个因果注意力块
- 损失：每个位置 1 个负样本的 BCE
- 论文参考成绩（ML-1M）：HR@10 ≈ 0.82，NDCG@10 ≈ 0.59

【运行方式】
    .venv/Scripts/python.exe src/train_sasrec.py --epochs 100
    消融示例：.venv/Scripts/python.exe src/train_sasrec.py --d 128 --blocks 3
"""
import argparse
import time
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


# ---------------------------------------------------------------- 数据
def load_sequences(min_len=5, data_dir=None, max_users=0, seed=42):
    """返回用户交互序列列表（物品下标从 1 开始，0 留给 padding）。

    支持两种数据集目录:
    - ml-1m : ratings.dat, '::' 分隔, 无表头
    - ml-25m: ratings.csv, ',' 分隔, 有表头 (userId,movieId,...)
    max_users > 0 时随机子采样用户（用于大数据集的快速验证）。
    """
    data_dir = Path(data_dir) if data_dir else DATA
    if (data_dir / "ratings.dat").exists():
        ratings = pd.read_csv(data_dir / "ratings.dat", sep="::",
                              engine="python",
                              names=["user_id", "movie_id", "rating",
                                     "timestamp"])
    else:
        ratings = pd.read_csv(data_dir / "ratings.csv")
        ratings = ratings.rename(columns={"userId": "user_id",
                                          "movieId": "movie_id"})
    i_map = {m: i + 1 for i, m in enumerate(sorted(ratings["movie_id"].unique()))}
    ratings["i"] = ratings["movie_id"].map(i_map)
    seqs = [g.sort_values("timestamp")["i"].tolist()
            for _, g in ratings.groupby("user_id")]
    seqs = [s for s in seqs if len(s) >= min_len]   # 论文的过滤标准
    if max_users and len(seqs) > max_users:
        idx = np.random.default_rng(seed).choice(len(seqs), max_users,
                                                 replace=False)
        seqs = [seqs[i] for i in idx]
    return seqs, len(i_map)


def pad_left(seq, L):
    return [0] * (L - len(seq)) + seq[-L:]


# ---------------------------------------------------------------- 模型
class SASRec(nn.Module):
    """单向（因果）Transformer 序列推荐模型。

    结构：物品Embedding + 位置Embedding → dropout
         → N × [因果自注意力 + 前馈网络]（残差 + LayerNorm）
    预测：最后一个位置的输出向量 · 候选物品Embedding = 得分
    """

    def __init__(self, n_items, d=64, max_len=200, n_blocks=2,
                 n_heads=1, dropout=0.2):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d)
        self.emb_drop = nn.Dropout(dropout)
        self.max_len = max_len
        self.attn_blocks = nn.ModuleList([
            nn.MultiheadAttention(d, n_heads, dropout=dropout,
                                  batch_first=True)
            for _ in range(n_blocks)])
        self.ffn_blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(d, d), nn.ReLU(),
                          nn.Dropout(dropout), nn.Linear(d, d))
            for _ in range(n_blocks)])
        self.norms1 = nn.ModuleList(
            [nn.LayerNorm(d) for _ in range(n_blocks)])
        self.norms2 = nn.ModuleList(
            [nn.LayerNorm(d) for _ in range(n_blocks)])
        self.final_norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def encode(self, seq):
        """seq: [B, L] 物品下标（左侧补 0）→ 每个位置的上下文向量 [B, L, d]"""
        B, L = seq.shape
        x = self.item_emb(seq) + self.pos_emb.weight[:L].unsqueeze(0)
        x = self.emb_drop(x)
        pad = (seq == 0)                                  # padding 位置掩码
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool,
                                       device=seq.device), diagonal=1)
        for attn, ffn, n1, n2 in zip(self.attn_blocks, self.ffn_blocks,
                                     self.norms1, self.norms2):
            # 因果掩码：位置 i 只能看 ≤i 的历史（不能偷看未来）
            a = attn(x, x, x, attn_mask=causal,
                     key_padding_mask=pad, need_weights=False)[0]
            x = n1(x + self.dropout(a))
            x = n2(x + self.dropout(ffn(x)))
        x = self.final_norm(x)
        x = x * (~pad).unsqueeze(-1)                      # padding 位置清零
        return x

    def score(self, ctx, item_ids):
        """上下文向量 · 物品 Embedding → 得分"""
        return (ctx * self.item_emb(item_ids)).sum(-1)


# ---------------------------------------------------------------- 评估
@torch.no_grad()
def evaluate(model, seqs, n_items, part="test", n_neg=100, K=10, n_eval=0):
    """论文协议：目标物品 + 100 个负采样，HR@K / NDCG@K。
    n_eval > 0 时随机子采样评估用户（大数据集快速验证用）。"""
    model.eval()
    rng = np.random.default_rng(42)
    if n_eval and len(seqs) > n_eval:
        idx = rng.choice(len(seqs), n_eval, replace=False)
        seqs = [seqs[i] for i in idx]
    hits, ndcgs = [], []
    for s in seqs:
        if part == "valid":
            hist, target = s[:-2], s[-2]
        else:
            hist, target = s[:-1], s[-1]
        L = min(len(hist), model.max_len)
        seq = torch.tensor([pad_left(hist, L)], device=DEVICE)
        ctx = model.encode(seq)[0, -1]                    # 最后位置 = "现在"
        negs = rng.integers(1, n_items + 1, n_neg)
        # 避免负采样撞到目标（论文做法：撞了重采）
        while target in negs:
            negs[negs == target] = rng.integers(1, n_items + 1)
        cand = torch.tensor(np.append(negs, target), device=DEVICE)
        rank = (model.score(ctx, cand) >
                model.score(ctx, cand[-1])).sum().item()  # 目标排在第几名
        hits.append(rank < K)
        ndcgs.append(1 / np.log2(rank + 2) if rank < K else 0.0)
    return float(np.mean(hits)), float(np.mean(ndcgs))


# ---------------------------------------------------------------- 训练
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--maxlen", type=int, default=200)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--dataset", choices=["ml1m", "ml25m"], default="ml1m",
                    help="ml1m=data/ml-1m; ml25m=data/ml-25m")
    ap.add_argument("--max_users", type=int, default=0,
                    help=">0 时子采样用户数（大数据集快速验证）")
    ap.add_argument("--eval_users", type=int, default=0,
                    help=">0 时子采样评估用户数")
    ap.add_argument("--ckpt", default="sasrec.pt",
                    help="checkpoint 文件名（避免覆盖其他数据集的权重）")
    ap.add_argument("--resume", action="store_true",
                    help="从 checkpoints/--ckpt 继续训练")
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = ExperimentConfig.from_args(__file__, args, seed=args.seed)
    cfg.save(EXP_DIR)
    print(f"实验配置已保存: {cfg.run_name}")
    data_dir = ROOT / "data" / ("ml-25m" if args.dataset == "ml25m" else "ml-1m")
    seqs, n_items = load_sequences(data_dir=data_dir,
                                   max_users=args.max_users)
    L = args.maxlen
    print(f"设备 {DEVICE} | 用户 {len(seqs)} 物品 {n_items} "
          f"| 交互 {sum(len(s) for s in seqs):,}")
    print(f"配置: d={args.d} blocks={args.blocks} maxlen={L} "
          f"dropout={args.dropout} lr={args.lr}")

    model = SASRec(n_items, args.d, L, args.blocks, dropout=args.dropout) \
        .to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           betas=(0.9, 0.98))
    bce = nn.BCEWithLogitsLoss(reduction="none")

    # 训练输入：序列 s[:-2]；每个位置 t 的正样本是 s[t+1]
    train_seqs = [s[:-2] for s in seqs]
    run = args.tag or f"d{args.d}-b{args.blocks}-L{L}-do{args.dropout}"
    log_path = EXP_DIR / "sasrec_log.csv"
    best_valid, best_state = 0.0, None
    start_epoch = 1
    ckpt_path = CKPT_DIR / args.ckpt
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, weights_only=True)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        start_epoch = ck.get("epoch", 0) + 1
        best_valid = ck.get("best_valid", 0.0)
        best_state = ck.get("best_state")
        print(f"从 epoch {start_epoch - 1} 续训（历史最佳 valid NDCG "
              f"{best_valid:.4f}）")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        t0, total, nb = time.time(), 0.0, 0
        order = np.random.permutation(len(train_seqs))
        for s0 in range(0, len(order), args.bs):
            batch = [train_seqs[j] for j in order[s0: s0 + args.bs]]
            # 输入 = 右移一位的序列；正样本 = 原序列
            inp = torch.tensor([pad_left([0] + b[:-1], L) for b in batch],
                               device=DEVICE)
            pos = torch.tensor([pad_left(b, L) for b in batch],
                               device=DEVICE)
            neg = torch.randint(1, n_items + 1, pos.shape, device=DEVICE)
            ctx = model.encode(inp)
            pos_score = model.score(ctx, pos)
            neg_score = model.score(ctx, neg)
            mask = (pos != 0).float()
            loss = (bce(pos_score, torch.ones_like(pos_score))
                    + bce(neg_score, torch.zeros_like(neg_score)))
            loss = (loss * mask).sum() / mask.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            v_hr, v_nd = evaluate(model, seqs, n_items, "valid",
                                  n_eval=args.eval_users)
            flag = ""
            if v_nd > best_valid:
                best_valid = v_nd
                best_state = {k: v.clone() for k, v
                              in model.state_dict().items()}
                flag = " ★ 目前最佳"
            print(f"epoch {epoch:3d} | loss {total / nb:.4f} "
                  f"| valid HR@10 {v_hr:.4f} NDCG@10 {v_nd:.4f} "
                  f"| {time.time() - t0:.1f}s{flag}")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{run},{epoch},{total / nb:.6f},{v_hr:.6f},"
                        f"{v_nd:.6f}\n")

    model.load_state_dict(best_state)
    t_hr, t_nd = evaluate(model, seqs, n_items, "test",
                          n_eval=args.eval_users)
    print(f"\n★ 测试集（验证集选优后）: HR@10 {t_hr:.4f} NDCG@10 {t_nd:.4f}")
    print(f"论文参考值: HR@10 ≈ 0.82, NDCG@10 ≈ 0.59")
    torch.save({"model": model.state_dict(), "args": vars(args),
                "opt": opt.state_dict(), "epoch": epoch,
                "best_valid": best_valid, "best_state": best_state},
               ckpt_path)


if __name__ == "__main__":
    main()
