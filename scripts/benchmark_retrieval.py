# -*- coding: utf-8 -*-
"""Rolling temporal full-catalog retrieval benchmark.

The benchmark keeps one protocol for every model:

* expanding training windows defined by global timestamps;
* a fixed future horizon after each cutoff;
* one first positive event per warm user in that horizon;
* every unseen catalog item is eligible (no sampled negatives);
* relevance, coverage, popularity and novelty are reported together;
* stochastic two-tower runs are repeated over explicit seeds.

Examples:
    .venv/Scripts/python.exe scripts/benchmark_retrieval.py `
        --tag baselines-v1 --models popular,item_knn
    .venv/Scripts/python.exe scripts/benchmark_retrieval.py `
        --tag two-tower-3seed-v1 --models popular,item_knn,two_tower `
        --seeds 42,43,44 --epochs 20
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import set_seed  # noqa: E402
from feedback import bootstrap_mean_interval  # noqa: E402
from train_two_tower import fit_two_tower, load_data  # noqa: E402


DEFAULT_TRAIN_FRACS = (0.70, 0.80, 0.90)
DEFAULT_SEEDS = (42, 43, 44)


def rolling_time_splits(
    ratings: pd.DataFrame,
    positives: pd.DataFrame,
    *,
    train_fracs: Iterable[float],
    horizon_frac: float,
) -> list[dict]:
    """Build expanding global-time windows with a fixed, non-leaking horizon."""
    if not 0.0 < horizon_frac < 1.0:
        raise ValueError("horizon_frac must be between 0 and 1")
    fractions = tuple(float(value) for value in train_fracs)
    if not fractions:
        raise ValueError("at least one train fraction is required")
    if any(not 0.0 < value < 1.0 for value in fractions):
        raise ValueError("train fractions must be between 0 and 1")
    if len(set(fractions)) != len(fractions):
        raise ValueError("train fractions must be unique")
    if any(value + horizon_frac > 1.0 for value in fractions):
        raise ValueError("train fraction + horizon fraction cannot exceed 1")

    windows: list[dict] = []
    for fraction in sorted(fractions):
        cutoff = float(ratings["timestamp"].quantile(fraction))
        horizon_end = float(
            ratings["timestamp"].quantile(fraction + horizon_frac))
        train = positives[positives["timestamp"] <= cutoff].copy()
        future = positives[
            (positives["timestamp"] > cutoff)
            & (positives["timestamp"] <= horizon_end)
        ].sort_values("timestamp")
        test_all = future.groupby("u", sort=False).head(1)
        warm_users = set(train["u"].unique())
        cold_users = int((~test_all["u"].isin(warm_users)).sum())
        test = test_all[test_all["u"].isin(warm_users)].copy()
        windows.append({
            "train_frac": fraction,
            "horizon_frac": horizon_frac,
            "cutoff_quantile_timestamp": cutoff,
            "horizon_end_quantile_timestamp": horizon_end,
            "train_max_timestamp": (
                int(train["timestamp"].max()) if len(train) else None),
            "test_min_timestamp": (
                int(test["timestamp"].min()) if len(test) else None),
            "test_max_timestamp": (
                int(test["timestamp"].max()) if len(test) else None),
            "train": train,
            "test": test,
            "cold_users_excluded": cold_users,
        })
    return windows


def popularity_order(train: pd.DataFrame, n_items: int) -> np.ndarray:
    counts = np.bincount(train["i"].to_numpy(), minlength=n_items)
    # Primary key: descending count. Secondary key: ascending item id.
    return np.lexsort((np.arange(n_items), -counts))


def popular_recommendations(
    train: pd.DataFrame,
    users: Iterable[int],
    *,
    n_items: int,
    k: int,
) -> dict[int, list[int]]:
    order = popularity_order(train, n_items)
    seen = train.groupby("u")["i"].apply(set).to_dict()
    return {
        int(user): [
            int(item) for item in order if item not in seen.get(user, set())][:k]
        for user in users
    }


def item_knn_neighbors(
    train: pd.DataFrame,
    *,
    n_users: int,
    n_items: int,
    neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return top cosine-cooccurrence neighbours for each item.

    ML-1M's 6040 x 3706 binary matrix is small enough for one dense BLAS
    multiplication. Only top neighbours are retained; the 55 MiB Gram matrix
    is released when this function returns.
    """
    width = min(max(1, neighbors), max(1, n_items - 1))
    matrix = np.zeros((n_users, n_items), dtype=np.float32)
    matrix[train["u"].to_numpy(), train["i"].to_numpy()] = 1.0
    counts = matrix.sum(axis=0)
    gram = matrix.T @ matrix
    denom = np.sqrt(np.maximum(counts[:, None] * counts[None, :], 1.0))
    gram /= denom
    np.fill_diagonal(gram, -np.inf)

    indices = np.argpartition(gram, -width, axis=1)[:, -width:]
    values = np.take_along_axis(gram, indices, axis=1)
    order = np.argsort(values, axis=1)[:, ::-1]
    return (
        np.take_along_axis(indices, order, axis=1).astype(np.int32),
        np.take_along_axis(values, order, axis=1).astype(np.float32),
    )


def item_knn_recommendations(
    train: pd.DataFrame,
    users: Iterable[int],
    *,
    n_users: int,
    n_items: int,
    k: int,
    neighbors: int,
    history_limit: int,
) -> dict[int, list[int]]:
    neighbor_ids, neighbor_scores = item_knn_neighbors(
        train, n_users=n_users, n_items=n_items, neighbors=neighbors)
    histories = (
        train.sort_values("timestamp").groupby("u")["i"].apply(list).to_dict())
    fallback = popularity_order(train, n_items)
    output: dict[int, list[int]] = {}
    for user in users:
        full_history = histories.get(user, [])
        # Recent context bounds scoring cost; the exclusion set must remain the
        # complete history or old interactions can leak back into recommendations.
        history = full_history[-history_limit:]
        seen = set(full_history)
        scores = np.zeros(n_items, dtype=np.float32)
        if history:
            candidates = neighbor_ids[history].reshape(-1)
            weights = neighbor_scores[history].reshape(-1)
            np.add.at(scores, candidates, weights)
        if seen:
            scores[np.fromiter(seen, dtype=np.int64)] = -np.inf
        candidate_order = np.argsort(scores)[::-1]
        recs = [
            int(item) for item in candidate_order
            if np.isfinite(scores[item]) and scores[item] > 0.0][:k]
        if len(recs) < k:
            selected = set(recs) | seen
            recs.extend(int(item) for item in fallback
                        if item not in selected)
        output[int(user)] = recs[:k]
    return output


@torch.no_grad()
def two_tower_recommendations(
    model,
    train: pd.DataFrame,
    users: Iterable[int],
    *,
    u_attr: np.ndarray,
    gid: np.ndarray,
    gmask: np.ndarray,
    n_items: int,
    k: int,
) -> dict[int, list[int]]:
    """Full-catalog exact inner-product ranking for selected users."""
    model.eval()
    device = next(model.parameters()).device
    user_ids = np.asarray(list(users), dtype=np.int64)
    item_vecs = model.item_vec(
        torch.arange(n_items, device=device),
        torch.as_tensor(gid, device=device),
        torch.as_tensor(gmask, device=device),
    ).cpu().numpy()
    user_vecs = model.user_vec(
        torch.as_tensor(user_ids, device=device),
        torch.as_tensor(u_attr, device=device),
    ).cpu().numpy()
    scores = user_vecs @ item_vecs.T
    seen = train.groupby("u")["i"].apply(set).to_dict()
    output = {}
    for row, user in enumerate(user_ids):
        user_seen = seen.get(int(user), set())
        if user_seen:
            scores[row, np.fromiter(user_seen, dtype=np.int64)] = -np.inf
        order = np.argsort(scores[row])[::-1][:k]
        output[int(user)] = [int(item) for item in order]
    return output


def evaluate_recommendations(
    recommendations: dict[int, list[int]],
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    n_items: int,
    k: int,
    bootstrap_samples: int,
) -> dict:
    """Compute relevance plus catalog-distribution metrics."""
    true_by_user = test.groupby("u")["i"].first().to_dict()
    seen_by_user = train.groupby("u")["i"].apply(set).to_dict()
    item_counts = np.bincount(train["i"].to_numpy(), minlength=n_items)
    head_n = max(1, math.ceil(n_items * 0.20))
    head = set(np.lexsort((np.arange(n_items), -item_counts))[:head_n])
    hits: list[float] = []
    ndcgs: list[float] = []
    reciprocal_ranks: list[float] = []
    exposed: list[int] = []
    for user, true_item in true_by_user.items():
        recs = recommendations.get(int(user), [])[:k]
        if len(recs) != k:
            raise ValueError(f"user {user} has {len(recs)} recommendations, expected {k}")
        if len(set(recs)) != len(recs):
            raise ValueError(f"user {user} has duplicate recommendations")
        if any(item < 0 or item >= n_items for item in recs):
            raise ValueError(f"user {user} has an out-of-catalog recommendation")
        leaked = set(recs) & seen_by_user.get(user, set())
        if leaked:
            raise ValueError(
                f"user {user} received already-seen items: {sorted(leaked)}")
        exposed.extend(recs)
        if true_item in recs:
            rank = recs.index(true_item)
            hits.append(1.0)
            ndcgs.append(1.0 / np.log2(rank + 2))
            reciprocal_ranks.append(1.0 / (rank + 1))
        else:
            hits.append(0.0)
            ndcgs.append(0.0)
            reciprocal_ranks.append(0.0)

    n = len(hits)
    if n == 0:
        raise ValueError("test window has no warm users")
    exposed_arr = np.asarray(exposed, dtype=np.int64)
    popularity = item_counts[exposed_arr] if len(exposed_arr) else np.zeros(0)
    smoothed_probability = (
        (popularity + 1.0) / (len(train) + n_items)
        if len(popularity) else np.zeros(0))
    recall_ci = bootstrap_mean_interval(hits, samples=bootstrap_samples)
    return {
        "test_users": n,
        f"recall@{k}": round(float(np.mean(hits)), 6),
        f"recall@{k}_ci95": [round(float(x), 6) for x in recall_ci],
        f"ndcg@{k}": round(float(np.mean(ndcgs)), 6),
        f"mrr@{k}": round(float(np.mean(reciprocal_ranks)), 6),
        f"catalog_coverage@{k}": round(
            len(set(exposed)) / n_items if n_items else 0.0, 6),
        f"long_tail_share@{k}": round(
            float(np.mean([item not in head for item in exposed]))
            if exposed else 0.0, 6),
        f"avg_log_popularity@{k}": round(
            float(np.mean(np.log1p(popularity))) if len(popularity) else 0.0, 6),
        f"novelty_bits@{k}": round(
            float(np.mean(-np.log2(smoothed_probability)))
            if len(smoothed_probability) else 0.0, 6),
        "candidate_catalog_size": n_items,
        "sampled_negatives": False,
    }


def aggregate_seed_runs(rows: list[dict], metric: str) -> list[dict]:
    grouped: dict[tuple[int, str], list[float]] = {}
    for row in rows:
        grouped.setdefault((row["window"], row["model"]), []).append(row[metric])
    return [
        {
            "window": window,
            "model": model,
            "runs": len(values),
            f"mean_{metric}": round(float(np.mean(values)), 6),
            f"std_{metric}": round(float(np.std(values)), 6),
            f"min_{metric}": round(float(np.min(values)), 6),
            f"max_{metric}": round(float(np.max(values)), 6),
        }
        for (window, model), values in sorted(grouped.items())
    ]


def summarize_across_windows(rows: list[dict], metric: str) -> list[dict]:
    """Macro-average windows after first averaging stochastic seeds per window."""
    per_window: dict[tuple[int, str], list[float]] = {}
    for row in rows:
        per_window.setdefault((row["window"], row["model"]), []).append(row[metric])
    per_model: dict[str, list[float]] = {}
    for (_window, model), values in per_window.items():
        per_model.setdefault(model, []).append(float(np.mean(values)))
    return [
        {
            "model": model,
            "windows": len(values),
            f"macro_mean_{metric}": round(float(np.mean(values)), 6),
            f"window_std_{metric}": round(float(np.std(values)), 6),
            f"worst_window_{metric}": round(float(np.min(values)), 6),
            f"best_window_{metric}": round(float(np.max(values)), 6),
        }
        for model, values in sorted(per_model.items())
    ]


def parse_numbers(raw: str, cast) -> tuple:
    return tuple(cast(value.strip()) for value in raw.split(",") if value.strip())


def reserve(tag: str) -> tuple[Path, Path]:
    json_path = ROOT / "experiments" / f"retrieval_benchmark_{tag}.json"
    csv_path = ROOT / "experiments" / f"retrieval_benchmark_{tag}.csv"
    existing = [str(path) for path in (json_path, csv_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite benchmark artifacts; choose a new --tag:\n"
            + "\n".join(existing))
    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--models", default="popular,item_knn,two_tower")
    parser.add_argument("--train-fracs", default="0.70,0.80,0.90")
    parser.add_argument("--horizon-frac", type=float, default=0.05)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--bs", type=int, default=1024)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--knn-neighbors", type=int, default=100)
    parser.add_argument("--history-limit", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()

    models = tuple(value.strip() for value in args.models.split(",") if value.strip())
    allowed = {"popular", "item_knn", "two_tower"}
    unknown = set(models) - allowed
    if unknown:
        parser.error(f"unknown models: {sorted(unknown)}")
    train_fracs = parse_numbers(args.train_fracs, float)
    seeds = parse_numbers(args.seeds, int)
    if "two_tower" in models and not seeds:
        parser.error("two_tower requires at least one seed")
    json_path, csv_path = reserve(args.tag)

    # Reuse the canonical catalog feature builder; only the returned split is
    # discarded because this benchmark owns the rolling protocol.
    _train0, _test0, u_attr, gid, gmask, maps = load_data(
        split="time", test_frac=0.1)
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
    rows: list[dict] = []
    protocol_windows: list[dict] = []
    started = time.perf_counter()
    for window_id, window in enumerate(windows, start=1):
        train, test = window["train"], window["test"]
        users = [int(value) for value in test["u"].tolist()]
        protocol_windows.append({
            key: value for key, value in window.items()
            if key not in ("train", "test")
        } | {"window": window_id, "train_pairs": len(train),
             "test_users": len(test)})
        print(
            f"[window {window_id}/{len(windows)}] train_frac="
            f"{window['train_frac']:.2f} train={len(train)} test={len(test)}",
            flush=True)

        deterministic = []
        if "popular" in models:
            deterministic.append(("popular", popular_recommendations(
                train, users, n_items=n_items, k=args.k)))
        if "item_knn" in models:
            print("  building Item-kNN full-catalog baseline ...", flush=True)
            deterministic.append(("item_knn", item_knn_recommendations(
                train, users, n_users=n_users, n_items=n_items, k=args.k,
                neighbors=args.knn_neighbors,
                history_limit=args.history_limit)))
        for model_name, recs in deterministic:
            metrics = evaluate_recommendations(
                recs, train=train, test=test, n_items=n_items, k=args.k,
                bootstrap_samples=args.bootstrap_samples)
            rows.append({
                "window": window_id, "train_frac": window["train_frac"],
                "model": model_name, "seed": "deterministic", **metrics})
            print(f"  {model_name:<10} Recall@{args.k}="
                  f"{metrics[f'recall@{args.k}']:.4f}", flush=True)

        if "two_tower" in models:
            for seed in seeds:
                set_seed(seed)
                print(f"  training two_tower seed={seed} ...", flush=True)
                model = fit_two_tower(
                    train, u_attr, gid, gmask, n_users, n_items,
                    epochs=args.epochs, lr=args.lr, bs=args.bs,
                    eval_fn=None, log_path=None, verbose=False)
                recs = two_tower_recommendations(
                    model, train, users, u_attr=u_attr, gid=gid, gmask=gmask,
                    n_items=n_items, k=args.k)
                metrics = evaluate_recommendations(
                    recs, train=train, test=test, n_items=n_items, k=args.k,
                    bootstrap_samples=args.bootstrap_samples)
                rows.append({
                    "window": window_id, "train_frac": window["train_frac"],
                    "model": "two_tower", "seed": seed, **metrics})
                print(f"  two_tower seed={seed} Recall@{args.k}="
                      f"{metrics[f'recall@{args.k}']:.4f}", flush=True)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    metric = f"recall@{args.k}"
    report = {
        "experiment": "rolling temporal full-catalog retrieval benchmark",
        "protocol": {
            "split": "expanding global-time train + fixed future horizon",
            "target": "first positive event per warm user in each horizon",
            "candidate_set": "all unseen MovieLens-1M items",
            "sampled_negatives": False,
            "windows": protocol_windows,
        },
        "config": vars(args) | {
            "models": list(models), "train_fracs": list(train_fracs),
            "seeds": list(seeds)},
        "runs": rows,
        "seed_summary": aggregate_seed_runs(rows, metric),
        "cross_window_summary": summarize_across_windows(rows, metric),
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    with json_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== seed summary ===")
    for row in report["seed_summary"]:
        print(f"  window={row['window']} {row['model']:<10} "
              f"n={row['runs']} mean={row[f'mean_{metric}']:.4f} "
              f"std={row[f'std_{metric}']:.4f}")
    print("\n=== cross-window macro summary ===")
    for row in report["cross_window_summary"]:
        print(f"  {row['model']:<10} mean="
              f"{row[f'macro_mean_{metric}']:.4f} "
              f"worst={row[f'worst_window_{metric}']:.4f} "
              f"best={row[f'best_window_{metric}']:.4f}")
    print(f"Report: {json_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
