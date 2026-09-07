# -*- coding: utf-8 -*-
"""Benchmark a TIGER-inspired Semantic-ID generator under the P1 protocol.

The defining TIGER mechanism is retained: content vectors -> residual Semantic
IDs -> autoregressive Transformer generation.  This controlled ML-1M baseline
uses hashed title/genre metadata and RQ-KMeans instead of SentenceT5 + RQ-VAE;
the report records that difference and must not be called an exact reproduction.

Example:
    .venv/Scripts/python.exe scripts/benchmark_generative_retrieval.py `
      --tag p2-tiger-lite-3seed-v1 --seeds 42,43,44 --epochs 3
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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
    aggregate_seed_runs,
    evaluate_recommendations,
    parse_numbers,
    rolling_time_splits,
    summarize_across_windows,
)
from config import set_seed  # noqa: E402
from generative_retrieval import (  # noqa: E402
    SemanticIDIndex,
    build_item_content_vectors,
    fit_semantic_generator,
    semantic_recommendations,
)
from train_two_tower import load_data  # noqa: E402


DEFAULT_BASELINE_REPORT = (
    ROOT / "experiments"
    / "retrieval_benchmark_p1-rolling-fullcatalog-3seed-v2.json")


def reserve(tag: str) -> tuple[Path, Path]:
    json_path = ROOT / "experiments" / f"generative_retrieval_{tag}.json"
    csv_path = ROOT / "experiments" / f"generative_retrieval_{tag}.csv"
    existing = [str(path) for path in (json_path, csv_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite generative benchmark artifacts; choose a new --tag:\n"
            + "\n".join(existing))
    return json_path, csv_path


def load_comparable_baseline(
    path: Path,
    *,
    train_fracs: tuple[float, ...],
    horizon_frac: float,
    k: int,
) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"baseline report not found: {path}")
    raw = path.read_bytes()
    report = json.loads(raw.decode("utf-8"))
    config = report["config"]
    mismatches = []
    if tuple(float(value) for value in config["train_fracs"]) != train_fracs:
        mismatches.append("train_fracs")
    if not np.isclose(float(config["horizon_frac"]), horizon_frac):
        mismatches.append("horizon_frac")
    if int(config["k"]) != k:
        mismatches.append("k")
    if report["protocol"].get("sampled_negatives") is not False:
        mismatches.append("sampled_negatives")
    if mismatches:
        raise ValueError(
            "baseline report uses an incompatible protocol: " + ", ".join(mismatches))
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "cross_window_summary": report["cross_window_summary"],
        "windows": report["protocol"]["windows"],
    }


def cold_item_metrics(
    recommendations: dict[int, list[int]],
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    k: int,
) -> dict:
    train_items = set(int(item) for item in train["i"].unique())
    cold_rows = test[~test["i"].isin(train_items)]
    hits = [
        int(int(row.i) in recommendations[int(row.u)][:k])
        for row in cold_rows.itertuples(index=False)
    ]
    return {
        "cold_target_users": len(hits),
        f"cold_item_recall@{k}": (
            round(float(np.mean(hits)), 6) if hits else None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--train-fracs", default="0.70,0.80,0.90")
    parser.add_argument("--horizon-frac", type=float, default=0.05)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--max-examples", type=int, default=60000)
    parser.add_argument("--history-items", type=int, default=20)
    parser.add_argument("--codebooks", default="64,64,64")
    parser.add_argument("--quantizer-seed", type=int, default=2026)
    parser.add_argument("--kmeans-iterations", type=int, default=25)
    parser.add_argument("--title-buckets", type=int, default=192)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--validation-frac", type=float, default=0.10,
                        help="last fraction of each outer training window selects epoch")
    parser.add_argument("--validation-examples", type=int, default=20000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--user-batch", type=int, default=16)
    parser.add_argument("--prefix-chunk", type=int, default=128)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--baseline-report", type=Path,
                        default=DEFAULT_BASELINE_REPORT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    train_fracs = parse_numbers(args.train_fracs, float)
    seeds = parse_numbers(args.seeds, int)
    codebook_sizes = parse_numbers(args.codebooks, int)
    if not seeds:
        parser.error("at least one seed is required")
    if args.epochs < 1 or args.bs < 1 or args.max_examples < 0:
        parser.error("epochs/bs must be positive and max-examples non-negative")
    if not 0.0 < args.validation_frac < 0.5:
        parser.error("validation-frac must be in (0, 0.5)")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is unavailable")
    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device)
    json_path, csv_path = reserve(args.tag)
    baseline = load_comparable_baseline(
        args.baseline_report, train_fracs=train_fracs,
        horizon_frac=args.horizon_frac, k=args.k)

    _train0, _test0, _u_attr, _gid, _gmask, maps = load_data(
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

    print("building content-only Semantic IDs ...", flush=True)
    content_vectors = build_item_content_vectors(
        ROOT / "data" / "ml-1m" / "movies.dat", maps["i_map"],
        title_buckets=args.title_buckets)
    semantic_index = SemanticIDIndex.fit(
        content_vectors, codebook_sizes, seed=args.quantizer_seed,
        iterations=args.kmeans_iterations)
    semantic_profile = semantic_index.profile()
    semantic_profile["codes_sha256"] = hashlib.sha256(
        semantic_index.codes.tobytes()).hexdigest()
    print(json.dumps(semantic_profile, ensure_ascii=False, indent=2), flush=True)

    rows: list[dict] = []
    training_runs: list[dict] = []
    protocol_windows: list[dict] = []
    started = time.perf_counter()
    for window_id, window in enumerate(windows, start=1):
        train, test = window["train"], window["test"]
        users = [int(value) for value in test["u"].tolist()]
        protocol_window = {
            key: value for key, value in window.items()
            if key not in ("train", "test")
        } | {"window": window_id, "train_pairs": len(train),
             "test_users": len(test)}
        expected = baseline["windows"][window_id - 1]
        for key in ("train_pairs", "test_users", "train_max_timestamp",
                    "test_min_timestamp", "test_max_timestamp"):
            if protocol_window[key] != expected[key]:
                raise ValueError(
                    f"window {window_id} differs from baseline for {key}: "
                    f"{protocol_window[key]} != {expected[key]}")
        protocol_windows.append(protocol_window)
        print(
            f"[window {window_id}/{len(windows)}] train={len(train)} "
            f"test={len(test)} device={device}", flush=True)
        for seed in seeds:
            set_seed(seed)
            run_started = time.perf_counter()
            inner_cutoff = float(train["timestamp"].quantile(
                1.0 - args.validation_frac))
            development = train[train["timestamp"] <= inner_cutoff]
            print(
                f"  selecting epoch tiger_lite seed={seed} "
                f"dev={len(development)} ...", flush=True)
            selection_model, selection_history, selected_epoch = fit_semantic_generator(
                development, semantic_index, seed=seed, epochs=args.epochs,
                batch_size=args.bs, lr=args.lr,
                max_history_items=args.history_items,
                max_examples=args.max_examples, d_model=args.d_model,
                n_heads=args.heads, n_layers=args.layers,
                dropout=args.dropout, device=device,
                validation_frame=train,
                validation_after_timestamp=inner_cutoff,
                validation_max_examples=args.validation_examples)
            del selection_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            set_seed(seed)
            print(
                f"  retraining full outer window for {selected_epoch} epoch(s) ...",
                flush=True)
            model, loss_history, _ = fit_semantic_generator(
                train, semantic_index, seed=seed, epochs=selected_epoch,
                batch_size=args.bs, lr=args.lr,
                max_history_items=args.history_items,
                max_examples=args.max_examples, d_model=args.d_model,
                n_heads=args.heads, n_layers=args.layers,
                dropout=args.dropout, device=device)
            recs = semantic_recommendations(
                model, train, users, semantic_index.codes, k=args.k,
                max_history_items=args.history_items,
                user_batch=args.user_batch, prefix_chunk=args.prefix_chunk)
            metrics = evaluate_recommendations(
                recs, train=train, test=test,
                n_items=len(semantic_index.codes), k=args.k,
                bootstrap_samples=args.bootstrap_samples)
            metrics.update(cold_item_metrics(
                recs, train, test, k=args.k))
            elapsed = round(time.perf_counter() - run_started, 1)
            rows.append({
                "window": window_id,
                "train_frac": window["train_frac"],
                "model": "tiger_lite_rqkmeans",
                "seed": seed,
                "selected_epoch": selected_epoch,
                "final_train_loss": loss_history[-1]["loss"],
                "elapsed_s": elapsed,
                **metrics,
            })
            training_runs.append({
                "window": window_id, "seed": seed,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "inner_validation_cutoff": inner_cutoff,
                "inner_development_pairs": len(development),
                "selected_epoch": selected_epoch,
                "selection_history": selection_history,
                "full_retrain_history": loss_history,
                "elapsed_s": elapsed,
            })
            print(
                f"  seed={seed} Recall@{args.k}="
                f"{metrics[f'recall@{args.k}']:.4f} "
                f"coverage={metrics[f'catalog_coverage@{args.k}']:.4f} "
                f"elapsed={elapsed:.1f}s", flush=True)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    metric = f"recall@{args.k}"
    report = {
        "experiment": "TIGER-lite Semantic-ID generative retrieval",
        "evidence_level": "controlled baseline; not an exact TIGER reproduction",
        "faithfulness": {
            "retained": [
                "content-derived hierarchical residual Semantic IDs",
                "collision disambiguation token for one-to-one item lookup",
                "Transformer encoder-decoder with autoregressive ID likelihood",
                "exact constrained scoring of every catalog Semantic ID",
            ],
            "substituted": (
                "MovieLens hashed title/genre/decade vectors + RQ-KMeans replace "
                "SentenceT5 content embeddings + learned RQ-VAE"),
        },
        "protocol": {
            "split": "expanding global-time train + fixed future horizon",
            "target": "first positive event per warm user in each horizon",
            "candidate_set": "all unseen MovieLens-1M items",
            "generation": "exact full-catalog autoregressive Semantic-ID log-probability",
            "model_selection": (
                "epoch selected by generative loss on the final 10% of each outer "
                "training window, then retrained from the same seed on the complete "
                "outer training window; outer future is evaluated once"),
            "sampled_negatives": False,
            "catalog_metadata_assumption": (
                "all item titles/genres are available; interaction labels remain time-bounded"),
            "windows": protocol_windows,
        },
        "config": vars(args) | {
            "baseline_report": str(args.baseline_report),
            "train_fracs": list(train_fracs), "seeds": list(seeds),
            "codebooks": list(codebook_sizes), "resolved_device": device,
        },
        "semantic_id_index": semantic_profile,
        "baseline_snapshot": baseline,
        "runs": rows,
        "training_runs": training_runs,
        "seed_summary": aggregate_seed_runs(rows, metric),
        "cross_window_summary": summarize_across_windows(rows, metric),
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    with json_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== generative cross-window summary ===")
    for row in report["cross_window_summary"]:
        print(
            f"  {row['model']} mean={row[f'macro_mean_{metric}']:.4f} "
            f"worst={row[f'worst_window_{metric}']:.4f} "
            f"best={row[f'best_window_{metric}']:.4f}")
    print(f"Report: {json_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
