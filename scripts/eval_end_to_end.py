# -*- coding: utf-8 -*-
"""召回→实验精排端到端联合评估：可选排序层到底有没有增量？

【为什么需要这个脚本】
项目此前分开报两个数字：双塔的 Recall@10 = 0.098，DeepFM 的 AUC = 0.754。
旧版线上链路曾是「双塔召回 50 → DeepFM 精排 → Top-10」。本脚本发现它显著
伤害 Recall 后，默认服务已切换为双塔 Top-K；这里保留显式精排路径用于回归、
失败复现和候选 ranker 晋升评估。

【本脚本回答三个问题】
  1. 召回天花板：真实物品有多大比例进得了 50 个候选？（Recall@N，精排无法
     挽回漏召回的部分，这是整条链路的上限）
  2. 端到端效果：精排完之后 Top-10 命中率是多少？（Recall@K / NDCG@K）
  3. **精排到底有没有用**：DeepFM 排序 vs 直接拿双塔的前 K 个，谁更好？
     这是项目此前完全没有验证过的一环——如果精排还不如召回自己的顺序，
     那这一层就是纯粹的延迟浪费。

【评估协议】
沿用 `src/train_two_tower.py` 的 leave-one-out：每个用户按时间的最后一条评分，
且该评分 ≥4 时作为待预测的正样本。已看过的物品由服务端自己剔除。
注意这与 DeepFM 的时间切分协议不同，两者不可直接比较（见 AGENTS.md）。

【复用线上路径】
候选与排序都取自 `Recommender.recommend(..., return_candidates=True)`，也就是
与 `serve.py` / `app.py` 同源的代码，避免评估脚本自己另写一套召回逻辑。

【运行】
    .venv/Scripts/python.exe scripts/eval_end_to_end.py --tag e2e-v1
    .venv/Scripts/python.exe scripts/eval_end_to_end.py --tag smoke --limit 200
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from serving import Recommender  # noqa: E402
from feedback import bootstrap_mean_interval  # noqa: E402
from train_two_tower import load_data as tt_load  # noqa: E402


def dcg_hit(position: int) -> float:
    """单个正样本命中在 0-based position 处的 NDCG 贡献（IDCG=1）。"""
    return 1.0 / np.log2(position + 2)


def rank_of(items: list[int], target: int, cutoff: int) -> int | None:
    """target 在 items 前 cutoff 个里的 0-based 位置；不在则 None。"""
    for position, item in enumerate(items[:cutoff]):
        if item == target:
            return position
    return None


def evaluate(rec: Recommender, test_pairs, *, k: int, n_candidates: int):
    """跑完整链路，同时统计召回上限、端到端效果和「精排是否有用」。"""
    stats = {
        "users": 0,
        "recall_stage_hits": 0,        # 真实物品进了候选池
        "end_to_end_hits": 0,          # 精排后进了 Top-K
        "recall_only_hits": 0,         # 不精排、直接取双塔前 K
        "end_to_end_ndcg": 0.0,
        "recall_only_ndcg": 0.0,
        "ranker_promoted": 0,          # 精排把它排得更靠前
        "ranker_demoted": 0,           # 精排把它排得更靠后
        "ranker_unchanged": 0,
        "empty_candidate_users": 0,
    }
    # 曝光集中度诊断：精排若退化成"热门度排序"，不同用户的 Top-K 会高度重合。
    # 这是「端到端变差」最常见的原因，光看 Recall 数字看不出来。
    retrieval_exposure: Counter[int] = Counter()
    ranked_exposure: Counter[int] = Counter()
    # 逐用户的命中向量：精排 vs 不精排是**同一批用户**的配对观测，
    # 直接相减再 bootstrap 才能回答"这个差距是真的还是噪声"。
    paired_delta: list[float] = []
    for user_id, true_movie_id in test_pairs:
        result = rec.recommend(user_id, k=n_candidates,
                               n_candidates=n_candidates,
                               return_candidates=True)
        recall_order = result["recall_order"]
        ranked_order = result["ranked_order"]
        if not recall_order:
            stats["empty_candidate_users"] += 1
            continue
        stats["users"] += 1
        retrieval_exposure.update(recall_order[:k])
        ranked_exposure.update(ranked_order[:k])

        # 1) 召回阶段：进没进候选池（这是整条链路的天花板）
        recall_position = rank_of(recall_order, true_movie_id, n_candidates)
        if recall_position is None:
            paired_delta.append(0.0)    # 漏召回：两条路径都必然未命中
            continue
        stats["recall_stage_hits"] += 1

        # 2) 端到端：精排后是否进 Top-K
        ranked_position = rank_of(ranked_order, true_movie_id, k)
        end_to_end_hit = float(ranked_position is not None)
        if ranked_position is not None:
            stats["end_to_end_hits"] += 1
            stats["end_to_end_ndcg"] += dcg_hit(ranked_position)

        # 3) 对照：不精排，直接用双塔顺序取前 K
        recall_only_hit = float(recall_position < k)
        if recall_position < k:
            stats["recall_only_hits"] += 1
            stats["recall_only_ndcg"] += dcg_hit(recall_position)
        paired_delta.append(end_to_end_hit - recall_only_hit)

        # 4) 精排把真实物品挪前还是挪后了（只对召回命中的用户统计）
        full_ranked_position = rank_of(
            ranked_order, true_movie_id, len(ranked_order))
        if full_ranked_position is None:
            continue
        if full_ranked_position < recall_position:
            stats["ranker_promoted"] += 1
        elif full_ranked_position > recall_position:
            stats["ranker_demoted"] += 1
        else:
            stats["ranker_unchanged"] += 1
    return stats, retrieval_exposure, ranked_exposure, paired_delta


def exposure_profile(counter: Counter, k: int, top_n: int = 20) -> dict:
    """曝光集中度：去重物品数 + 头部占比。数字越集中，个性化越弱。"""
    total = sum(counter.values())
    if not total:
        return {"distinct_items": 0, f"top{top_n}_share": None}
    return {
        "distinct_items": len(counter),
        f"top{top_n}_share": round(
            sum(count for _, count in counter.most_common(top_n)) / total, 6),
        "most_exposed": [
            {"movie_id": movie_id,
             "user_share": round(count / (total / k), 6)}
            for movie_id, count in counter.most_common(5)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True, help="产物标签；已存在则拒绝覆盖")
    parser.add_argument("--k", type=int, default=10, help="最终展示条数")
    parser.add_argument("--n-candidates", type=int, default=50,
                        help="召回候选池大小（线上默认 50）")
    parser.add_argument("--limit", type=int, default=0,
                        help="只评估前 N 个用户（0 = 全部，用于冒烟）")
    parser.add_argument("--deepfm-ckpt", default=None,
                        help="替代精排权重路径；默认 checkpoints/deepfm.pt。"
                             "召回与协议完全不变，便于逐版本对比 ranker")
    args = parser.parse_args()
    if args.k <= 0 or args.n_candidates < args.k:
        parser.error("--k 必须为正，且不得大于 --n-candidates")

    output = ROOT / "experiments" / f"eval_end_to_end_{args.tag}.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}; choose a new --tag")

    print("Loading serving pipeline (two-tower + DeepFM + Faiss) ...", flush=True)
    if args.deepfm_ckpt:
        print(f"  ranker override: {args.deepfm_ckpt}", flush=True)
    started = time.perf_counter()
    rec = Recommender.load(
        ROOT, deepfm_ckpt=args.deepfm_ckpt, ranking_policy="deepfm")
    print(f"  loaded in {time.perf_counter() - started:.1f}s", flush=True)

    # leave-one-out 测试集：与 train_two_tower.py 逐字对齐
    _train, test, *_rest, maps = tt_load()
    idx2movie = {v: k for k, v in maps["i_map"].items()}
    idx2user = {v: k for k, v in maps["u_map"].items()}
    test_by_user = test.groupby("u")["i"].first()
    pairs = [(int(idx2user[u]), int(idx2movie[i]))
             for u, i in test_by_user.items()]
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"Evaluating {len(pairs)} leave-one-out users "
          f"(recall {args.n_candidates} -> rank {args.k})", flush=True)

    evaluation_started = time.perf_counter()
    stats, retrieval_exposure, ranked_exposure, paired_delta = evaluate(
        rec, pairs, k=args.k, n_candidates=args.n_candidates)
    elapsed = time.perf_counter() - evaluation_started

    n = max(1, stats["users"])
    recall_ceiling = stats["recall_stage_hits"] / n
    end_to_end_recall = stats["end_to_end_hits"] / n
    recall_only = stats["recall_only_hits"] / n
    retrieval_profile = exposure_profile(retrieval_exposure, args.k)
    ranked_profile = exposure_profile(ranked_exposure, args.k)
    # 配对 bootstrap：单位是用户，两条路径在同一批用户上观测，相减消掉用户
    # 难度的差异。与 `src/feedback.py` 的 OPE 决策门用的是同一个估计器。
    delta_interval = bootstrap_mean_interval(paired_delta, samples=2000)
    if delta_interval is None:
        significance = "insufficient_data"
    elif delta_interval[0] > 0.0:
        significance = "ranking_helps_significantly"
    elif delta_interval[1] < 0.0:
        significance = "ranking_hurts_significantly"
    else:
        significance = "indistinguishable_from_no_ranking"
    report = {
        "experiment": "end-to-end retrieval -> ranking evaluation",
        "protocol": {
            "split": "leave-one-out (each user's last rating, kept when >= 4)",
            "note": ("not comparable with the DeepFM time-split AUC; "
                     "different dataset split and different task"),
            "k": args.k,
            "n_candidates": args.n_candidates,
            "path": ("Recommender.recommend with explicit "
                     "ranking_policy=deepfm"),
            "ranking_policy": "deepfm (experimental; not the service default)",
            "ranker_checkpoint": args.deepfm_ckpt or "checkpoints/deepfm.pt",
        },
        "users_evaluated": stats["users"],
        "users_with_empty_candidates": stats["empty_candidate_users"],
        "retrieval_ceiling": {
            f"recall@{args.n_candidates}": round(recall_ceiling, 6),
            "meaning": "真实物品进入候选池的比例；精排无法挽回漏召回的部分",
        },
        "end_to_end": {
            f"recall@{args.k}": round(end_to_end_recall, 6),
            f"ndcg@{args.k}": round(stats["end_to_end_ndcg"] / n, 6),
        },
        "retrieval_only_baseline": {
            f"recall@{args.k}": round(recall_only, 6),
            f"ndcg@{args.k}": round(stats["recall_only_ndcg"] / n, 6),
            "meaning": "跳过 DeepFM，直接取双塔前 K —— 精排必须赢过这个才有价值",
        },
        "ranker_effect": {
            "recall@k_delta": round(end_to_end_recall - recall_only, 6),
            "recall@k_delta_ci95": (
                [round(bound, 6) for bound in delta_interval]
                if delta_interval else None),
            "significance": significance,
            "promoted_true_item": stats["ranker_promoted"],
            "demoted_true_item": stats["ranker_demoted"],
            "unchanged": stats["ranker_unchanged"],
            "verdict": (
                "ranking_helps" if end_to_end_recall > recall_only
                else "ranking_hurts" if end_to_end_recall < recall_only
                else "ranking_neutral"),
        },
        "funnel": {
            "stage_1_retrieval": round(recall_ceiling, 6),
            "stage_2_ranking_kept": (
                round(end_to_end_recall / recall_ceiling, 6)
                if recall_ceiling else None),
            "meaning": ("stage_2 = 召回命中的物品里，有多大比例被精排留在 "
                        f"Top-{args.k}"),
        },
        "exposure_concentration": {
            "retrieval_top_k": retrieval_profile,
            "ranked_top_k": ranked_profile,
            "diagnosis": (
                "精排后去重物品数明显下降 + 头部占比明显上升 = 排序层退化成"
                "热门度排序，个性化被抹平。pointwise 正向评分目标在全库样本上"
                "训练，学到的主要是物品质量先验；线上却只对召回选出的高相关"
                "候选打分，这个分布它没见过（sample selection bias）。"),
        },
        "elapsed_s": round(elapsed, 2),
        "ms_per_user": round(elapsed / n * 1000, 3),
    }
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("\n========== 召回 → 精排 漏斗 ==========")
    print(f"  召回天花板 Recall@{args.n_candidates:<3d} = {recall_ceiling:.4f}"
          f"   ({stats['recall_stage_hits']}/{stats['users']} 进了候选池)")
    print(f"  端到端    Recall@{args.k:<3d} = {end_to_end_recall:.4f}"
          f"   NDCG@{args.k} = {stats['end_to_end_ndcg'] / n:.4f}")
    print(f"  仅召回    Recall@{args.k:<3d} = {recall_only:.4f}"
          f"   NDCG@{args.k} = {stats['recall_only_ndcg'] / n:.4f}"
          f"   ← 不用精排的对照")
    print(f"\n  精排增量 ΔRecall@{args.k} = "
          f"{end_to_end_recall - recall_only:+.4f}  "
          f"[{report['ranker_effect']['verdict']}]")
    print(f"  精排把真实物品：前移 {stats['ranker_promoted']} 次 / "
          f"后移 {stats['ranker_demoted']} 次 / "
          f"不变 {stats['ranker_unchanged']} 次")
    if recall_ceiling:
        print(f"\n  漏斗：召回命中的物品里 "
              f"{end_to_end_recall / recall_ceiling:.1%} 被精排留在 Top-{args.k}")
    print(f"\n========== 曝光集中度（Top-{args.k}）==========")
    print(f"  双塔召回顺序 : 去重物品 {retrieval_profile['distinct_items']:4d} 个"
          f"，前 20 占曝光 {retrieval_profile['top20_share']:.1%}")
    print(f"  DeepFM 精排后: 去重物品 {ranked_profile['distinct_items']:4d} 个"
          f"，前 20 占曝光 {ranked_profile['top20_share']:.1%}")
    if (ranked_profile["distinct_items"]
            < retrieval_profile["distinct_items"]):
        # 纯 ASCII 前缀：Windows 中文控制台默认 GBK，emoji/符号会直接抛
        # UnicodeEncodeError 把脚本打断。
        print("  [warn] 精排收敛到更少的物品上 —— 排序层正在把个性化换成热门度")
    print(f"\nReport: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
