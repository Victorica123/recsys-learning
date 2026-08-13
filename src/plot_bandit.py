# -*- coding: utf-8 -*-
"""阶段 4：Bandit 三种策略的在线 CTR 曲线
用 Kimi Work 托管 Python 运行：python src/plot_bandit.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent


def main():
    df = pd.read_csv(ROOT / "experiments" / "bandit_log.csv")

    setup_plot()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.lineplot(data=df, x="round", y="running_ctr", hue="policy", ax=ax,
                 linewidth=2,
                 palette={"随机": "#999999", "ε-greedy": "#4C72B0",
                          "LinUCB": "#C44E52"})
    ax.axhline(0.677, color="#4C72B0", linestyle="--", alpha=0.6)
    ax.text(3000, 0.682, "0.677 = 不看人的策略的天花板（全局最优单一类型）",
            color="#4C72B0", fontsize=10)
    ax.set_title("探索-利用实验：LinUCB 靠个性化突破了非个性化天花板")
    ax.set_xlabel("推荐轮数")
    ax.set_ylabel("在线 CTR（累计平均）")
    out = ROOT / "figures" / "09-bandit探索利用.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("已保存:", out)


if __name__ == "__main__":
    main()
