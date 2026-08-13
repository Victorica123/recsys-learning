# -*- coding: utf-8 -*-
"""模块2 图表: PG训练曲线 / 五策略长期收益对比 / DQN v1-v2 稳定性对比。
用托管 Python 运行: python src/plot_pg.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from daimon_runtime import setup_plot

setup_plot()

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)


# ---------------------------------------------------------- 图10: PG 训练曲线
pg = pd.read_csv(EXP / "pg_log.csv")
fig, ax1 = plt.subplots(figsize=(8.5, 4.6))
ax1.plot(pg["update"], pg["avg_return"], color="tab:blue", lw=1.2, alpha=0.55)
ax1.plot(pg["update"], pg["avg_return"].rolling(7, min_periods=1).mean(),
         color="tab:blue", lw=2.8, label="平均回合奖励(7点滑动平均)")
ax1.axhline(6.69, color="gray", ls="--", lw=1.2)
ax1.text(pg["update"].max() * 0.02, 6.69 + 0.25, "随机基线 6.69", color="gray",
         fontsize=9)
ax1.axhline(2.81, color="tab:red", ls="--", lw=1.2)
ax1.text(pg["update"].max() * 0.02, 2.81 + 0.25, "贪心基线 2.81",
         color="tab:red", fontsize=9)
ax1.axvline(200, color="purple", ls=":", lw=1.2)
ax1.text(202, ax1.get_ylim()[0] + 0.5, "← 200次后熵退火", color="purple",
         fontsize=9)
ax1.set_xlabel("训练更新次数")
ax1.set_ylabel("平均回合奖励", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax2 = ax1.twinx()
ax2.plot(pg["update"], pg["entropy"], color="tab:orange", lw=1.6,
         label="策略熵(右轴)")
ax2.set_ylabel("策略熵", color="tab:orange")
ax2.tick_params(axis="y", labelcolor="tab:orange")
ax2.set_ylim(0, 6)
ax1.set_title("REINFORCE + Actor-Critic 训练曲线（奖励与策略熵）")
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="center right", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "10-PG训练曲线.png", bbox_inches="tight", dpi=130)
plt.close(fig)

# ---------------------------------------------------------- 图11: 五策略对比
pg_ev = pd.read_csv(EXP / "pg_eval.csv")
dqn_ev = pd.read_csv(EXP / "dqn_v2_eval.csv")

names = ["贪心(短视)", "策略梯度\n(argmax部署)", "随机", "Double DQN\n(确定性)",
         "策略梯度\n(采样部署)"]
keys = [("pg", "贪心(短视)"), ("pg", "策略梯度(argmax)"), ("pg", "随机"),
        ("dqn", "Double DQN"), ("pg", "策略梯度(采样)")]


def get(src, key):
    df = pg_ev if src == "pg" else dqn_ev
    r = df[df["policy"] == key].iloc[0]
    return float(r["avg_reward"]), float(r["avg_len"]), float(r["genre_diversity"])


rewards, lens, divs = zip(*[get(s, k) for s, k in keys])
colors = ["#d62728", "#ff7f0e", "#7f7f7f", "#1f77b4", "#2ca02c"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
b1 = axes[0].bar(names, rewards, color=colors, alpha=0.85)
for rect, v in zip(b1, rewards):
    axes[0].text(rect.get_x() + rect.get_width() / 2, v + 0.25, f"{v:.2f}",
                 ha="center", fontsize=10, fontweight="bold")
axes[0].set_ylabel("平均会话总奖励")
axes[0].set_title("长期收益：RL 策略 vs 基线（300 会话）")
axes[0].set_ylim(0, max(rewards) * 1.18)
axes[0].tick_params(axis="x", labelsize=9)

b2 = axes[1].bar(names, divs, color=colors, alpha=0.85)
for rect, v, L in zip(b2, divs, lens):
    axes[1].text(rect.get_x() + rect.get_width() / 2, v + 0.1,
                 f"{v:.2f}\n(长{L:.1f})", ha="center", fontsize=9)
axes[1].set_ylabel("平均类型多样性（会话内不同类型数）")
axes[1].set_title("推荐多样性与会话长度")
axes[1].set_ylim(0, max(divs) * 1.3)
axes[1].tick_params(axis="x", labelsize=9)
fig.tight_layout()
fig.savefig(FIG / "11-长期收益对比.png", bbox_inches="tight", dpi=130)
plt.close(fig)

# ---------------------------------------------------------- 图12: DQN v1 vs v2
v1 = pd.read_csv(EXP / "dqn_v1_log.csv", header=None,
                 names=["step", "eps", "q_mean"])
v2 = pd.read_csv(EXP / "dqn_v2_log.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].plot(v1["step"], v1["q_mean"], color="tab:red", lw=1.8)
axes[0].set_title("v1（软目标更新+弱环境信号）：训练失败", fontsize=11)
axes[0].set_xlabel("环境步数")
axes[0].set_ylabel("回放批次 Q 均值")
axes[0].annotate("Q 值震荡无学习方向\n评估 4.12 < 随机 7.21 < 贪心 9.43",
                 xy=(v1["step"].iloc[-1] * 0.42, v1["q_mean"].max() * 0.75),
                 fontsize=9, color="tab:red")
axes[1].plot(v2["step"], v2["q_mean"], color="tab:green", lw=1.8)
axes[1].set_title("v2（硬目标同步+LayerNorm+新环境）：稳定收敛", fontsize=11)
axes[1].set_xlabel("环境步数")
axes[1].annotate("Q 平稳爬升（理论上限≈9）\n评估 7.63 > 随机 6.69 >> 贪心 2.81",
                 xy=(v2["step"].iloc[-1] * 0.05, v2["q_mean"].max() * 0.72),
                 fontsize=9, color="tab:green")
fig.suptitle("手写 DQN 的失败与修复：同一算法、不同工程决策", fontsize=12)
fig.tight_layout()
fig.savefig(FIG / "12-DQN稳定性对比.png", bbox_inches="tight", dpi=130)
plt.close(fig)

print("已生成:")
for p in ["10-PG训练曲线.png", "11-长期收益对比.png", "12-DQN稳定性对比.png"]:
    fp = FIG / p
    print(f"  {fp}  ({fp.stat().st_size // 1024} KB)")
