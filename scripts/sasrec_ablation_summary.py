# -*- coding: utf-8 -*-
"""SASRec ablation matrix: re-evaluate checkpoints and write summary + figures.

Loads the main model and the four one-at-a-time ablation checkpoints, re-runs
the paper leave-one-out test protocol (target + 100 negatives, full user set),
and merges the best-valid rows from ``experiments/sasrec_log.csv`` into
``experiments/sasrec_ablation_matrix.csv`` plus two figures:

* ``figures/sasrec_ablation_valid_ndcg-<date>.png`` - valid NDCG training curves
* ``figures/sasrec_ablation_test-<date>.png`` - test HR@10 / NDCG@10 bars

Run: ``.venv/Scripts/python.exe scripts/sasrec_ablation_summary.py``
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from train_sasrec import SASRec, evaluate, load_sequences  # noqa: E402


CKPT_DIR = ROOT / "checkpoints"
EXP_DIR = ROOT / "experiments"
FIG_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data" / "ml-1m"


def best_valid_from_log() -> dict[str, dict]:
    """Best valid HR/NDCG per run tag from experiments/sasrec_log.csv."""
    log_path = EXP_DIR / "sasrec_log.csv"
    if not log_path.exists():
        return {}
    best: dict[str, dict] = {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        parts = line.split(",")
        if len(parts) < 5:
            continue
        tag, epoch = parts[0], int(parts[1])
        hr, ndcg = float(parts[3]), float(parts[4])
        entry = best.setdefault(
            tag, {"epoch": epoch, "hr": hr, "ndcg": ndcg})
        if ndcg > entry["ndcg"]:
            entry.update(epoch=epoch, hr=hr, ndcg=ndcg)
    return best


def main() -> None:
    seqs, n_items = load_sequences(data_dir=DATA_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_best = best_valid_from_log()

    runs = [
        ("baseline", CKPT_DIR / "sasrec_main.pt", "d64-b2-L200-do0.2"),
        ("d128", CKPT_DIR / "sasrec_ablation_d128.pt", "ablation-d128"),
        ("b3", CKPT_DIR / "sasrec_ablation_b3.pt", "ablation-b3"),
        ("L50", CKPT_DIR / "sasrec_ablation_L50.pt", "ablation-L50"),
        ("do05", CKPT_DIR / "sasrec_ablation_do05.pt", "ablation-do05"),
    ]

    summary = []
    for name, ckpt_path, log_tag in runs:
        if not ckpt_path.exists():
            print(f"skip missing checkpoint: {ckpt_path.name}")
            continue
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        args = ck["args"]
        model = SASRec(
            n_items, args["d"], args["maxlen"], args["blocks"],
            dropout=args["dropout"],
        )
        model.load_state_dict(ck["model"])
        model = model.to(device)
        hr, ndcg = evaluate(model, seqs, n_items, "test")
        best = log_best.get(log_tag, {})
        summary.append({
            "variant": name,
            "d": args["d"],
            "blocks": args["blocks"],
            "maxlen": args["maxlen"],
            "dropout": args["dropout"],
            "best_valid_epoch": best.get("epoch"),
            "best_valid_hr10": round(float(best.get("hr", 0.0)), 4),
            "best_valid_ndcg10": round(float(best.get("ndcg", 0.0)), 4),
            "test_hr10": round(hr, 4),
            "test_ndcg10": round(ndcg, 4),
        })
        print(
            f"{name:>8} d={args['d']} b={args['blocks']} L={args['maxlen']} "
            f"do={args['dropout']} | valid NDCG {summary[-1]['best_valid_ndcg10']} "
            f"@ ep {summary[-1]['best_valid_epoch']} | "
            f"test HR@10 {hr:.4f} NDCG@10 {ndcg:.4f}")

    today = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d")
    csv_path = EXP_DIR / "sasrec_ablation_matrix.csv"
    if csv_path.exists():
        raise FileExistsError(f"refusing to overwrite {csv_path}")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(summary[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary)
    print(f"matrix: {csv_path}")

    # ---- valid NDCG curves ----
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"baseline": "#1f77b4", "d128": "#d62728", "b3": "#2ca02c",
              "L50": "#ff7f0e", "do05": "#9467bd"}
    for name, _, log_tag in runs:
        tag_map = {"baseline": "d64-b2-L200-do0.2"}.get(log_tag, log_tag)
        xs, ys = [], []
        log_path = EXP_DIR / "sasrec_log.csv"
        if not log_path.exists():
            continue
        for line in log_path.read_text(encoding="utf-8").splitlines():
            parts = line.split(",")
            if len(parts) < 5 or parts[0] != tag_map:
                continue
            xs.append(int(parts[1]))
            ys.append(float(parts[4]))
        if xs:
            order = sorted(range(len(xs)), key=lambda i: xs[i])
            xs = [xs[i] for i in order]
            ys = [ys[i] for i in order]
            ax.plot(xs, ys, marker="o", ms=3, label=name, color=colors[name])
    ax.set_xlabel("epoch")
    ax.set_ylabel("valid NDCG@10")
    ax.set_title("SASRec ablation: valid NDCG@10 (ML-1M, LOO + 100 negatives)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    curve_path = FIG_DIR / f"sasrec_ablation_valid_ndcg-{today}.png"
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)
    print(f"figure: {curve_path}")

    # ---- test metric bars ----
    variants = [row["variant"] for row in summary]
    hr = [row["test_hr10"] for row in summary]
    ndcg = [row["test_ndcg10"] for row in summary]
    x = np.arange(len(variants))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, hr, width, label="HR@10", color="#4c72b0")
    bars2 = ax.bar(x + width / 2, ndcg, width, label="NDCG@10", color="#dd8452")
    for bars in (bars1, bars2):
        for bar in bars:
            ax.annotate(
                f"{bar.get_height():.4f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center", va="bottom", fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_ylim(0, max(max(hr), max(ndcg)) * 1.18)
    ax.set_ylabel("test metric")
    ax.set_title("SASRec ablation: test HR@10 / NDCG@10 (ML-1M)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    bar_path = FIG_DIR / f"sasrec_ablation_test-{today}.png"
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    print(f"figure: {bar_path}")


if __name__ == "__main__":
    main()
