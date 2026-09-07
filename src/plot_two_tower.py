# -*- coding: utf-8 -*-
"""阶段 3：双塔训练曲线 + 召回方法对比图
用 Kimi Work 托管 Python 运行：python src/plot_two_tower.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from plotting import setup_plot

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent


def main():
    df = pd.read_csv(ROOT / "experiments" / "two_tower_log.csv",
                     names=["model", "epoch", "loss", "recall", "ndcg"])

    setup_plot()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.lineplot(data=df, x="epoch", y="recall", ax=axes[0], color="#4C72B0",
                 linewidth=2)
    axes[0].set_title("双塔模型训练过程（Recall@10）")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("Recall@10")

    cmp_df = pd.DataFrame({
        "方法": ["矩阵分解\n(阶段1)", "双塔·未校正\n(踩坑版)", "双塔·logit校正\n(工业做法)"],
        "Recall@10": [0.062, 0.044, 0.097]})
    sns.barplot(data=cmp_df, x="方法", y="Recall@10", ax=axes[1],
                color="#55A868")
    for i, v in enumerate(cmp_df["Recall@10"]):
        axes[1].text(i, v + 0.002, f"{v:.3f}", ha="center")
    axes[1].set_title("召回方法对比：评估口径和采样偏差修正的力量")
    axes[1].set_ylim(0, 0.12)

    out = ROOT / "figures" / "08-双塔召回对比.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("已保存:", out)


if __name__ == "__main__":
    main()
