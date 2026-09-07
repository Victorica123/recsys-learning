# -*- coding: utf-8 -*-
"""
RL 快速体验：PPO 训练奖励曲线

用 Kimi Work 托管 Python 运行：python src/plot_ppo_curve.py
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
    df = pd.read_csv(ROOT / "experiments" / "ppo_log.csv",
                     names=["steps", "reward", "length"])
    df["reward_smooth"] = df["reward"].rolling(50, min_periods=1).mean()

    setup_plot()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=df, x="steps", y="reward", ax=ax,
                 alpha=0.25, color="#4C72B0", label="单回合奖励（原始）")
    sns.lineplot(data=df, x="steps", y="reward_smooth", ax=ax,
                 color="#C44E52", linewidth=2, label="50回合滑动平均")
    ax.axhline(200, color="green", linestyle="--", alpha=0.7)
    ax.text(df["steps"].max() * 0.02, 210, "200 = 官方判定'学会着陆'",
            color="green")
    ax.set_title("PPO 学登月：奖励从 -100（坠毁）一路爬到 +250（稳稳降落）")
    ax.set_xlabel("训练步数")
    ax.set_ylabel("回合奖励")
    out = ROOT / "figures" / "06-PPO训练奖励曲线.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("已保存:", out)


if __name__ == "__main__":
    main()
