# -*- coding: utf-8 -*-
"""Evaluate multi-route RRF and an honest rolling sequence-aware listwise ranker.

For window t, the residual ranker may train only on candidate/label groups
logged by windows < t. This prevents fitting on the evaluation window or using
a newer base model to recreate older candidates after it has seen their data.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmark_retrieval import (  # noqa: E402
    evaluate_recommendations,
    item_knn_recommendations,
    parse_numbers,
    popular_recommendations,
    rolling_time_splits,
    two_tower_recommendations,
)
from config import set_seed  # noqa: E402
from multistage_retrieval import (  # noqa: E402
    FEATURE_NAMES,
    ListwiseGroup,
    ResidualListwiseRanker,
    fit_residual_listwise_ranker,
    reciprocal_rank_fusion,
    sequence_candidate_features,
    validate_prior_window_groups,
)
from train_two_tower import fit_two_tower, load_data  # noqa: E402


def reserve(tag: str) -> Path:
    path = ROOT / "experiments" / f"multiroute_ranker_{tag}.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}; choose a new --tag")
    return path


def catalog_genres(gid: np.ndarray, gmask: np.ndarray) -> list[set[int]]:
    return [
        {int(genre) for genre, keep in zip(row, mask) if keep > 0}
        for row, mask in zip(gid, gmask)
    ]


def build_groups(
    *,
    window_id: int,
    train: pd.DataFrame,
    test: pd.DataFrame,
    routes_by_user: dict[int, dict[str, list[int]]],
    item_genres: list[set[int]],
    n_items: int,
    candidate_k: int,
    history_limit: int,
    rrf_constant: float,
) -> tuple[list[ListwiseGroup], dict[int, list[int]]]:
    histories = (
        train.sort_values("timestamp").groupby("u")["i"].apply(list).to_dict())
    counts = np.bincount(train["i"].to_numpy(), minlength=n_items)
    head_n = max(1, int(np.ceil(n_items * 0.20)))
    head = set(np.lexsort((np.arange(n_items), -counts))[:head_n])
    long_tail = set(range(n_items)) - head
    test_rows = test.set_index("u")
    groups: list[ListwiseGroup] = []
    rrf_top10: dict[int, list[int]] = {}
    for user, routes in routes_by_user.items():
        candidates, raw_scores = reciprocal_rank_fusion(
            routes, rrf_constant=rrf_constant, limit=candidate_k)
        max_score = max(float(raw_scores.max()), 1e-12)
        base_scores = raw_scores / max_score
        target = int(test_rows.loc[user, "i"])
        target_index = candidates.index(target) if target in candidates else None
        features = sequence_candidate_features(
            candidates, routes=routes,
            recent_items=histories.get(user, [])[-history_limit:],
            item_genres=item_genres, item_counts=counts,
            long_tail_items=long_tail)
        groups.append(ListwiseGroup(
            candidate_ids=np.asarray(candidates, dtype=np.int64),
            base_scores=base_scores.astype(np.float32),
            features=features,
            target_index=target_index,
            user_id=user,
            event_timestamp=int(test_rows.loc[user, "timestamp"]),
            window=window_id,
        ))
        rrf_top10[user] = candidates[:10]
    return groups, rrf_top10


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--train-fracs", default="0.70,0.80,0.90")
    parser.add_argument("--horizon-frac", type=float, default=0.05)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--bs", type=int, default=1024)
    parser.add_argument("--candidate-k", type=int, default=200)
    parser.add_argument("--knn-neighbors", type=int, default=100)
    parser.add_argument("--history-limit", type=int, default=100)
    parser.add_argument("--rrf-constant", type=float, default=60.0)
    parser.add_argument("--ranker-epochs", type=int, default=100)
    parser.add_argument("--ranker-lr", type=float, default=0.03)
    parser.add_argument("--ranker-l2", type=float, default=0.01)
    parser.add_argument("--residual-scale", type=float, default=0.20)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    output = reserve(args.tag)

    train_fracs = parse_numbers(args.train_fracs, float)
    seeds = parse_numbers(args.seeds, int)
    if len(train_fracs) < 2:
        parser.error("the rolling ranker requires at least two train windows")
    if not seeds:
        parser.error("at least one seed is required")
    if args.candidate_k < 10:
        parser.error("--candidate-k must be at least 10")

    _a, _b, u_attr, gid, gmask, maps = load_data(split="time", test_frac=0.1)
    ratings = pd.read_csv(
        ROOT / "data" / "ml-1m" / "ratings.dat", sep="::", engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"])
    ratings["u"] = ratings["user_id"].map(maps["u_map"])
    ratings["i"] = ratings["movie_id"].map(maps["i_map"])
    positives = ratings[ratings["rating"] >= 4]
    windows = rolling_time_splits(
        ratings, positives, train_fracs=train_fracs,
        horizon_frac=args.horizon_frac)
    n_users, n_items = len(u_attr), len(maps["i_map"])
    genres = catalog_genres(gid, gmask)

    # Deterministic routes are reused by all stochastic two-tower seeds.
    deterministic: dict[int, dict[str, dict[int, list[int]]]] = {}
    for window_id, window in enumerate(windows, start=1):
        train, test = window["train"], window["test"]
        users = [int(user) for user in test["u"]]
        print(f"precompute window={window_id} deterministic routes ...", flush=True)
        deterministic[window_id] = {
            "popular": popular_recommendations(
                train, users, n_items=n_items, k=args.candidate_k),
            "item_knn": item_knn_recommendations(
                train, users, n_users=n_users, n_items=n_items,
                k=args.candidate_k, neighbors=args.knn_neighbors,
                history_limit=args.history_limit),
        }

    results: list[dict] = []
    ranker_fits: list[dict] = []
    started = time.perf_counter()
    candidate_metric = f"candidate_recall@{args.candidate_k}"
    for seed in seeds:
        prior_groups: list[ListwiseGroup] = []
        for window_id, window in enumerate(windows, start=1):
            train, test = window["train"], window["test"]
            users = [int(user) for user in test["u"]]
            set_seed(seed)
            print(f"seed={seed} window={window_id}: train TwoTower ...", flush=True)
            model = fit_two_tower(
                train, u_attr, gid, gmask, n_users, n_items,
                epochs=args.epochs, lr=args.lr, bs=args.bs,
                eval_fn=None, log_path=None, verbose=False)
            tt = two_tower_recommendations(
                model, train, users, u_attr=u_attr, gid=gid, gmask=gmask,
                n_items=n_items, k=args.candidate_k)
            routes_by_user = {
                user: {
                    "two_tower": tt[user],
                    "item_knn": deterministic[window_id]["item_knn"][user],
                    "popular": deterministic[window_id]["popular"][user],
                }
                for user in users
            }
            groups, rrf_recs = build_groups(
                window_id=window_id, train=train, test=test,
                routes_by_user=routes_by_user, item_genres=genres,
                n_items=n_items, candidate_k=args.candidate_k,
                history_limit=args.history_limit,
                rrf_constant=args.rrf_constant)
            candidate_recall = float(np.mean(
                [group.target_index is not None for group in groups]))
            rrf_metrics = evaluate_recommendations(
                rrf_recs, train=train, test=test, n_items=n_items, k=10,
                bootstrap_samples=args.bootstrap_samples)
            results.append({
                "seed": seed, "window": window_id, "model": "rrf",
                "trained_on_windows": [],
                candidate_metric: round(candidate_recall, 6),
                **rrf_metrics,
            })

            if prior_groups:
                validate_prior_window_groups(prior_groups, window_id)
                ranker, fit_summary = fit_residual_listwise_ranker(
                    prior_groups, epochs=args.ranker_epochs,
                    lr=args.ranker_lr, l2=args.ranker_l2,
                    residual_scale=args.residual_scale, seed=seed)
                learned_recs = {
                    group.user_id: ranker.rank(group, 10) for group in groups}
                learned_metrics = evaluate_recommendations(
                    learned_recs, train=train, test=test, n_items=n_items, k=10,
                    bootstrap_samples=args.bootstrap_samples)
                trained_windows = sorted({group.window for group in prior_groups})
                results.append({
                    "seed": seed, "window": window_id,
                    "model": "sequence_residual_listwise",
                    "trained_on_windows": trained_windows,
                    candidate_metric: round(candidate_recall, 6),
                    **learned_metrics,
                })
                ranker_fits.append({
                    "seed": seed, "evaluation_window": window_id,
                    "trained_on_windows": trained_windows,
                    **fit_summary,
                })
                print(
                    f"  RRF={rrf_metrics['recall@10']:.4f} "
                    f"ranker={learned_metrics['recall@10']:.4f} "
                    f"candidate@{args.candidate_k}={candidate_recall:.4f}", flush=True)
            else:
                # Explicitly record why there is no learned W1 number.
                ranker_fits.append({
                    "seed": seed, "evaluation_window": window_id,
                    "trained_on_windows": [], "status": "no_prior_window"})
                print(
                    f"  RRF={rrf_metrics['recall@10']:.4f}; "
                    "ranker skipped (no prior window)", flush=True)

            # Only after evaluating window t may its honest candidates/labels
            # enter the training pool for t+1.
            prior_groups.extend(
                group for group in groups if group.target_index is not None)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = []
    for model_name in sorted({row["model"] for row in results}):
        values = [row["recall@10"] for row in results if row["model"] == model_name]
        summary.append({
            "model": model_name, "runs": len(values),
            "mean_recall@10": round(float(np.mean(values)), 6),
            "std_recall@10": round(float(np.std(values)), 6),
        })
    comparable_results = [row for row in results if row["window"] > 1]
    comparable_summary = []
    for model_name in ("rrf", "sequence_residual_listwise"):
        values = [row["recall@10"] for row in comparable_results
                  if row["model"] == model_name]
        comparable_summary.append({
            "model": model_name, "runs": len(values), "windows": [2, 3],
            "mean_recall@10": round(float(np.mean(values)), 6),
            "std_recall@10": round(float(np.std(values)), 6),
        })
    indexed = {
        (row["seed"], row["window"], row["model"]): row["recall@10"]
        for row in results
    }
    paired_deltas = [
        indexed[(seed, window, "sequence_residual_listwise")]
        - indexed[(seed, window, "rrf")]
        for seed in seeds for window in range(2, len(windows) + 1)
    ]
    report = {
        "experiment": "multi-route retrieval + rolling sequence residual listwise ranker",
        "protocol": {
            "base": "same rolling full-catalog windows as retrieval benchmark v2",
            "routes": ["two_tower", "item_knn", "popular"],
            "fusion": "weighted reciprocal rank fusion (equal weights)",
            "ranker_training": (
                "evaluation window t uses only honest candidate/label groups "
                "logged in windows < t"),
            "feature_names": list(FEATURE_NAMES),
        },
        "config": vars(args) | {
            "train_fracs": list(train_fracs), "seeds": list(seeds)},
        "results": results,
        "ranker_fits": ranker_fits,
        "summary": summary,
        "comparable_windows_summary": comparable_summary,
        "paired_ranker_minus_rrf": {
            "pairs": len(paired_deltas),
            "mean_delta_recall@10": round(float(np.mean(paired_deltas)), 6),
            "min_delta_recall@10": round(float(np.min(paired_deltas)), 6),
            "max_delta_recall@10": round(float(np.max(paired_deltas)), 6),
            "inference": "descriptive only; three seeds and two windows",
        },
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print("\n=== summary ===")
    for row in summary:
        print(f"  {row['model']:<28} n={row['runs']} "
              f"mean={row['mean_recall@10']:.4f} std={row['std_recall@10']:.4f}")
    print("=== comparable windows 2..N ===")
    for row in comparable_summary:
        print(f"  {row['model']:<28} n={row['runs']} "
              f"mean={row['mean_recall@10']:.4f}")
    print(f"  paired ranker-RRF Δ={np.mean(paired_deltas):+.4f} "
          "(descriptive)")
    print(f"Report: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
