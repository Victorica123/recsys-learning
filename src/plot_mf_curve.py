# -*- coding: utf-8 -*-
"""
阶段 1：把 experiments/mf_log.csv 画成训练曲线对比图

用 Kimi Work 托管 Python 运行（自带 seaborn 和中文字体）：
    python src/plot_mf_curve.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

RUN_LABELS = {
    "dim32-lr0.01-wd0.0": "实验A：无正则化 lr=0.01",
    "dim32-lr0.005-wd1e-05": "实验B：L2正则 lr=0.005",
}


def main():
    df = pd.read_csv(
        ROOT / "experiments" / "mf_log.csv",
        names=["run", "time", "epoch", "loss", "recall", "ndcg", "secs"],
    )
    df["run"] = df["run"].map(lambda r: RUN_LABELS.get(r, r))

    setup_plot()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.lineplot(data=df, x="epoch", y="loss", hue="run", ax=axes[0])
    axes[0].set_title("训练损失：两组都在持续下降")
    axes[0].set_xlabel("训练轮数 epoch")
    axes[0].set_ylabel("MSE 损失")

    sns.lineplot(data=df, x="epoch", y="recall", hue="run", ax=axes[1])
    axes[1].set_title("Recall@10：无正则化的过拟合崩溃了")
    axes[1].set_xlabel("训练轮数 epoch")
    axes[1].set_ylabel("Recall@10")

    out = ROOT / "figures" / "04-矩阵分解训练曲线.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("已保存:", out)


if __name__ == "__main__":
    main()
