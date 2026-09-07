# -*- coding: utf-8 -*-
"""
阶段 0：数据探索 —— 建立推荐系统的数据直觉

运行方式：python src/explore_data.py
产出：控制台统计信息 + figures/ 目录下的三张图

读完输出后，请打开 notes/01-数据探索.md 写下你的观察。
"""
import sys
from pathlib import Path

# 让脚本可以使用 Kimi Work 托管运行时里的绘图辅助（中文字体等）
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from plotting import setup_plot

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "ml-1m"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)


def load_data():
    """MovieLens 用 :: 分隔，且没有表头，需要手动指定列名。"""
    ratings = pd.read_csv(
        DATA / "ratings.dat", sep="::", engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )
    movies = pd.read_csv(
        DATA / "movies.dat", sep="::", engine="python",
        names=["movie_id", "title", "genres"], encoding="latin-1",
    )
    users = pd.read_csv(
        DATA / "users.dat", sep="::", engine="python",
        names=["user_id", "gender", "age", "occupation", "zip"],
    )
    return ratings, movies, users


def main():
    ratings, movies, users = load_data()

    # ---------- 1. 基本规模 ----------
    print("=" * 50)
    print("数据规模")
    print("=" * 50)
    print(f"用户数：{ratings['user_id'].nunique()}")
    print(f"电影数：{ratings['movie_id'].nunique()}")
    print(f"评分数：{len(ratings)}")
    # 稀疏度 = 1 - 实际评分 / 所有可能的用户-电影组合
    sparsity = 1 - len(ratings) / (ratings['user_id'].nunique()
                                   * ratings['movie_id'].nunique())
    print(f"评分矩阵稀疏度：{sparsity:.2%}  ← 推荐系统的核心难题！")

    setup_plot()  # 配置中文字体与后端，此后开始画图

    # ---------- 2. 评分分布 ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=ratings, x="rating", ax=ax, color="#4C72B0")
    ax.set_title("评分分布：用户更倾向打高分还是低分？")
    ax.set_xlabel("评分（星）")
    ax.set_ylabel("数量")
    fig.savefig(FIG / "01-评分分布.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------- 3. 每用户评分数量：长尾分布 ----------
    per_user = ratings.groupby("user_id").size()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(per_user, bins=50, ax=ax, color="#55A868")
    ax.set_title("每个用户评了多少部电影？（长尾分布 = 数据稀疏的来源）")
    ax.set_xlabel("单用户评分数量")
    ax.set_ylabel("用户个数")
    fig.savefig(FIG / "02-用户活跃度分布.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------- 4. 电影类型 Top10 ----------
    genre_counts = (
        movies.assign(genre=movies["genres"].str.split("|"))
        .explode("genre")["genre"].value_counts().head(10)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=genre_counts.values, y=genre_counts.index, ax=ax,
                color="#C44E52")
    ax.set_title("电影数量最多的 10 个类型")
    ax.set_xlabel("电影数量")
    ax.set_ylabel("类型")
    fig.savefig(FIG / "03-电影类型Top10.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\n图表已保存到 figures/ ：")
    print("  01-评分分布.png / 02-用户活跃度分布.png / 03-电影类型Top10.png")
    print("\n下一步：打开 notes/01-数据探索.md，写下你的观察。")


if __name__ == "__main__":
    main()
