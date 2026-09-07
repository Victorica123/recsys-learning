# -*- coding: utf-8 -*-
"""评估协议对照：leave-one-out 到底把 Recall 高估了多少？

【为什么需要这个脚本】
`train_two_tower.py` 默认用 leave-one-out（每个用户按时间的最后一条做测试）。
这个协议有一个已知缺陷：**时间穿越**——用户 A 的测试时刻可能早于用户 B 的
某些训练交互，模型等于用"未来的全局信息"预测"过去"。学术界 2020 年后对
leave-one-out 的主要批评就是这条，它会系统性高估 Recall。

本项目此前只报 leave-one-out 下的 0.098。本脚本把同一套架构在**全局时间切分**
（所有交互按时间排序，前 90% 训练、后 10% 测试，无穿越）下再跑一遍，把
"协议乐观偏差"从一句定性提醒变成一个可复查的数字。

【必须说清的三点边界】
1. 两组数字来自**两个模型 + 两个协议**，不是同一个模型换协议评估——因为
   leave-one-out 训练出来的模型见过测试期的其它交互，直接拿去评时间切分
   测试集本身就是泄漏。所以这是**协议级对照**，不是"同一模型掉了多少"。
2. 时间切分的训练数据更少（520829 vs 571729 对），任务也更难（预测更远的
   未来），所以差值里既有"消除穿越"的部分，也有"数据更少/任务更难"的部分，
   两者本脚本无法拆开。
3. 时间切分的测试用户只有 1157 个（leave-one-out 是 3552），方差明显更大，
   因此必须报 bootstrap 置信区间，不能只报点估计。

【运行】
    .venv/Scripts/python.exe src/train_two_tower.py --split time --epochs 20 `
        --ckpt checkpoints/two_tower_timesplit.pt
    .venv/Scripts/python.exe scripts/eval_split_protocols.py --tag protocol-v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from checkpoint_io import load_torch_checkpoint  # noqa: E402
from feedback import bootstrap_mean_interval  # noqa: E402
from train_two_tower import TwoTower, load_data  # noqa: E402


def per_user_hits(model, train, test, u_attr, gid, gmask, maps, k=10):
    """返回逐用户的命中向量（1/0）和 NDCG 向量，供 bootstrap 使用。

    检索逻辑与 `train_two_tower.evaluate` 完全一致：缓冲区按最大历史长度取，
    过滤已看，再截前 k 个。
    """
    import faiss
    model.eval()
    device = next(model.parameters()).device
    n_items = len(maps["i_map"])
    with torch.no_grad():
        item_vecs = model.item_vec(
            torch.arange(n_items, device=device),
            torch.as_tensor(gid, device=device),
            torch.as_tensor(gmask, device=device)).cpu().numpy()
        user_vecs = model.user_vec(
            torch.arange(len(u_attr), device=device),
            torch.as_tensor(u_attr, device=device)).cpu().numpy()

    index = faiss.IndexFlatIP(item_vecs.shape[1])
    index.add(item_vecs.astype(np.float32))
    seen = train.groupby("u")["i"].apply(set).to_dict()
    max_seen = max((len(items) for items in seen.values()), default=0)
    _, topk = index.search(user_vecs.astype(np.float32),
                           min(k + max_seen, n_items))

    test_by_user = test.groupby("u")["i"].first()
    hits, ndcgs = [], []
    for u, true_i in test_by_user.items():
        recs = [i for i in topk[u] if i >= 0 and i not in seen.get(u, set())][:k]
        if true_i in recs:
            hits.append(1.0)
            ndcgs.append(1.0 / np.log2(recs.index(true_i) + 2))
        else:
            hits.append(0.0)
            ndcgs.append(0.0)
    return hits, ndcgs


def evaluate_protocol(split, checkpoint, *, k, bootstrap_samples, test_frac):
    path = ROOT / checkpoint
    if not path.exists():
        raise FileNotFoundError(
            f"missing checkpoint {path}; train it first "
            f"(see this script's docstring)")
    train, test, u_attr, gid, gmask, maps = load_data(
        split=split, test_frac=test_frac)
    payload = load_torch_checkpoint(path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TwoTower(payload["n_users"], payload["n_items"]).to(device)
    model.load_state_dict(payload["model"])

    hits, ndcgs = per_user_hits(
        model, train, test, u_attr, gid, gmask, maps, k=k)
    return {
        "split": split,
        "checkpoint": checkpoint,
        "trained_on_split": payload.get("split", "loo (legacy checkpoint)"),
        "train_pairs": int(len(train)),
        "test_users": len(hits),
        "cold_users_excluded": int(maps["cold_users_excluded"]),
        f"recall@{k}": round(float(np.mean(hits)), 6),
        f"recall@{k}_ci95": [
            round(bound, 6) for bound in
            bootstrap_mean_interval(hits, samples=bootstrap_samples)],
        f"ndcg@{k}": round(float(np.mean(ndcgs)), 6),
        f"ndcg@{k}_ci95": [
            round(bound, 6) for bound in
            bootstrap_mean_interval(ndcgs, samples=bootstrap_samples)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True, help="产物标签；已存在则拒绝覆盖")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--loo-ckpt", default="checkpoints/two_tower.pt")
    parser.add_argument("--time-ckpt",
                        default="checkpoints/two_tower_timesplit.pt")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()

    output = ROOT / "experiments" / f"eval_split_protocols_{args.tag}.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    results = []
    for split, checkpoint in (("loo", args.loo_ckpt),
                              ("time", args.time_ckpt)):
        print(f"Evaluating split={split} ({checkpoint}) ...", flush=True)
        results.append(evaluate_protocol(
            split, checkpoint, k=args.k,
            bootstrap_samples=args.bootstrap_samples,
            test_frac=args.test_frac))

    loo, timed = results
    recall_key = f"recall@{args.k}"
    ratio = (timed[recall_key] / loo[recall_key]) if loo[recall_key] else None
    overlap = not (loo[f"{recall_key}_ci95"][0] > timed[f"{recall_key}_ci95"][1]
                   or timed[f"{recall_key}_ci95"][0] > loo[f"{recall_key}_ci95"][1])
    report = {
        "experiment": "evaluation protocol comparison (leave-one-out vs global time split)",
        "question": "how much does leave-one-out inflate Recall@K?",
        "caveats": [
            "two models under two protocols, not one model re-evaluated: a "
            "leave-one-out model has seen interactions from the test period, "
            "so evaluating it on the time-split test set would itself leak",
            "the time-split model trains on less data and faces a harder task "
            "(predicting further into the future); the gap mixes 'no time "
            "travel' with 'less data / harder task' and this script cannot "
            "separate them",
            "the time-split test set is much smaller, so its interval is wide",
        ],
        "results": results,
        "comparison": {
            "time_over_loo_ratio": round(ratio, 4) if ratio else None,
            "relative_drop": round(1 - ratio, 4) if ratio else None,
            "intervals_overlap": overlap,
            "reading": (
                "intervals overlap; the drop is directional but not "
                "separated by this sample size"
                if overlap else
                "intervals do not overlap; the protocol gap is larger than "
                "sampling noise at this sample size"),
        },
    }
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("\n========== 评估协议对照 ==========")
    for row in results:
        interval = row[f"{recall_key}_ci95"]
        print(f"  {row['split']:5s} | 训练对 {row['train_pairs']:6d} "
              f"| 测试用户 {row['test_users']:5d} "
              f"| Recall@{args.k} {row[recall_key]:.4f} "
              f"[{interval[0]:.4f}, {interval[1]:.4f}]")
        if row["cold_users_excluded"]:
            print(f"          （另有 {row['cold_users_excluded']} 个冷启动用户被剔除）")
    if ratio:
        print(f"\n  时间切分 / leave-one-out = {ratio:.1%}"
              f"（相对下降 {1 - ratio:.1%}）")
    print(f"  区间是否重叠：{'是' if overlap else '否'} —— {report['comparison']['reading']}")
    print(f"\nReport: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
