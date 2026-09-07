# -*- coding: utf-8 -*-
"""阶段 2b 图表：推荐系统"后训练"（SFT → 离线 RL）结果可视化。

生成两张图：
  13-离线RL后训练对比.png   各策略平均会话奖励（SFT 天花板 vs 离线 RL 越过）
  14-CQL保守系数权衡.png    CQL 保守系数 α 的奖励/多样性权衡曲线

运行（项目根目录，已装 matplotlib）：
    .venv/Scripts/python.exe src/plot_offline_rl.py
"""
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plotting import setup_plot

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)


setup_plot()


def reward_of(df, policy):
    return float(df.loc[df["policy"] == policy, "avg_reward"].iloc[0])


# ============================================================ 图13：策略对比
ev = pd.read_csv(EXP / "offline_rl_phase2_eval.csv")

# 讲故事的顺序：短视/SFT 家族 → CQL → 行为策略 → 随机 → naive 离线 RL
order = ["短视贪心", "SFT 行为克隆", "奖励加权 BC", "CQL 离线 RL",
         "行为策略(ε-greedy短视)", "随机策略", "naive 离线 DQN"]
labels = ["短视贪心\n(行为策略本质)", "SFT\n行为克隆", "奖励加权\nBC",
          "CQL\n离线RL", "行为策略\n(ε-greedy)", "随机\n(交错基线)",
          "naive 离线\nDQN"]
rewards = [reward_of(ev, p) for p in order]
colors = ["#9aa0a6", "#5b8ff9", "#5b8ff9", "#ff9845",
          "#b0b0b0", "#7f7f7f", "#2ca02c"]

ceiling = reward_of(ev, "SFT 行为克隆")   # SFT ≈ 行为策略天花板

fig, ax = plt.subplots(figsize=(10, 5.2))
bars = ax.bar(labels, rewards, color=colors, alpha=0.9, width=0.66)
for rect, v in zip(bars, rewards):
    ax.text(rect.get_x() + rect.get_width() / 2, v + 0.12, f"{v:.2f}",
            ha="center", fontsize=11, fontweight="bold")
ax.axhline(ceiling, color="#5b8ff9", ls="--", lw=1.3)
ax.annotate("SFT 天花板 ≈ 行为策略\n（模仿学不出更好）",
            xy=(2.5, ceiling), xytext=(0.15, ceiling + 3.2),
            color="#3b6fd4", fontsize=10,
            arrowprops=dict(arrowstyle="->", color="#3b6fd4", lw=1.2))
ax.annotate("离线 RL（Bellman）越过天花板\n学出交错策略，奖励 3× ↑",
            xy=(6, rewards[-1]), xytext=(3.4, rewards[-1] + 0.6),
            fontsize=10, color="#1f7a1f",
            arrowprops=dict(arrowstyle="->", color="#1f7a1f", lw=1.4))
ax.set_ylabel("平均会话总奖励（200 会话）")
ax.set_title("推荐系统后训练：SFT 复刻行为策略，离线 RL 越过其天花板\n"
             "（RecSimEnv 仿真会话，含疲劳/厌倦动力学）", fontsize=12)
ax.set_ylim(0, max(rewards) * 1.2)
ax.tick_params(axis="x", labelsize=9.5)
fig.tight_layout()
fig.savefig(FIG / "13-离线RL后训练对比.png", bbox_inches="tight", dpi=130)
plt.close(fig)


# ============================================================ 图14：α 权衡
sw = pd.read_csv(EXP / "offline_rl_alpha_sweep.csv")
sw["cql_alpha"] = pd.to_numeric(sw["cql_alpha"], errors="coerce")
cql = sw.dropna(subset=["cql_alpha"]).sort_values("cql_alpha")
alphas = cql["cql_alpha"].tolist()
rew = cql["avg_reward"].tolist()
div = cql["genre_diversity"].tolist()
rand_ref = reward_of(sw, "随机策略")
greedy_ref = reward_of(sw, "短视贪心")

fig, ax1 = plt.subplots(figsize=(9.2, 5.0))
xs = list(range(len(alphas)))
xt = [("naive\nα=0" if a == 0 else f"CQL\nα={a:g}") for a in alphas]
l1 = ax1.plot(xs, rew, "o-", color="#2ca02c", lw=2.4, ms=8,
              label="平均会话奖励（左轴）")
for x, v in zip(xs, rew):
    ax1.text(x, v + 0.25, f"{v:.2f}", ha="center", fontsize=10,
             color="#1f7a1f", fontweight="bold")
ax1.axhline(rand_ref, color="gray", ls="--", lw=1.1)
ax1.text(len(xs) - 1.6, rand_ref + 0.15, f"随机基线 {rand_ref:.2f}",
         color="gray", fontsize=9)
ax1.axhline(greedy_ref, color="#d62728", ls="--", lw=1.1)
ax1.text(len(xs) - 1.9, greedy_ref + 0.15, f"短视贪心 {greedy_ref:.2f}",
         color="#d62728", fontsize=9)
ax1.set_xticks(xs)
ax1.set_xticklabels(xt)
ax1.set_xlabel("保守系数 α（0 = 纯离线 DQN，越大越保守）")
ax1.set_ylabel("平均会话奖励", color="#2ca02c")
ax1.tick_params(axis="y", labelcolor="#2ca02c")
ax1.set_ylim(0, max(rew) * 1.18)

ax2 = ax1.twinx()
l2 = ax2.plot(xs, div, "s--", color="#ff7f0e", lw=2.0, ms=7,
              label="类型多样性（右轴）")
ax2.set_ylabel("会话内类型多样性", color="#ff7f0e")
ax2.tick_params(axis="y", labelcolor="#ff7f0e")
ax2.set_ylim(0, max(div) * 1.25)

ax1.annotate("α↑ 越保守 → 策略被拉回短视行为分布\n（多样性塌到 ≈1.0）",
             xy=(len(xs) - 1, div[-1]), xytext=(1.2, max(rew) * 0.55),
             fontsize=10, color="#a0620a",
             arrowprops=dict(arrowstyle="->", color="#a0620a", lw=1.3))
lns = l1 + l2
ax1.legend(lns, [x.get_label() for x in lns], loc="upper right", fontsize=9.5)
ax1.set_title("CQL 保守系数 α 的权衡：本仿真器覆盖足够，保守只减分\n"
              "（先诊断数据覆盖/回报结构，再决定是否上保守）", fontsize=12)
fig.tight_layout()
fig.savefig(FIG / "14-CQL保守系数权衡.png", bbox_inches="tight", dpi=130)
plt.close(fig)

print("已生成:")
for p in ["13-离线RL后训练对比.png", "14-CQL保守系数权衡.png"]:
    fp = FIG / p
    print(f"  {fp}  ({fp.stat().st_size // 1024} KB)")
