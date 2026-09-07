# -*- coding: utf-8 -*-
"""阶段 2：LR / FM / DeepFM 训练曲线对比图
用 Kimi Work 托管 Python 运行：python src/plot_deepfm_curve.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from plotting import setup_plot

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
LABELS = {"lr": "LR 逻辑回归", "fm": "FM 因子分解机", "deepfm": "DeepFM"}


def main():
    df = pd.read_csv(ROOT / "experiments" / "deepfm_log.csv",
                     names=["model", "epoch", "loss", "valid_auc"])
    df["model"] = df["model"].map(LABELS)

    setup_plot()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.lineplot(data=df, x="epoch", y="loss", hue="model", ax=axes[0])
    axes[0].set_title("训练损失（BCE 交叉熵）")
    axes[0].set_xlabel("epoch")
    sns.lineplot(data=df, x="epoch", y="valid_auc", hue="model", ax=axes[1])
    axes[1].set_title("验证集 AUC：特征交叉（FM）带来质变")
    axes[1].set_xlabel("epoch")
    axes[1].axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    out = ROOT / "figures" / "07-CTR模型对比.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("已保存:", out)


if __name__ == "__main__":
    main()
