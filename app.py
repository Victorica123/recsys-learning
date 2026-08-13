# -*- coding: utf-8 -*-
"""
阶段 5：推荐系统可交互 Demo（工业界"召回 → 排序"两段式流水线）

任选一个用户，实时演示：
  1) 双塔模型 + Faiss 从 3706 部电影中召回 50 个候选（毫秒级）
  2) DeepFM 对 50 个候选逐个打"喜欢概率"，精排出 Top-10

流水线核心复用 `src/serving.py` 的 `Recommender`（与 REST API `serve.py` 同源），
本文件只负责 Streamlit 展示层。

运行方式（在项目根目录）：
    .venv/Scripts/python.exe -m streamlit run app.py
然后浏览器打开 http://localhost:8501
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from serving import Recommender, check_artifacts  # noqa: E402  可复用推荐核心

st.set_page_config(page_title="智能推荐 Demo", page_icon="🎬", layout="wide")


@st.cache_resource
def get_recommender():
    """加载数据 + 两个模型（只在第一次运行时执行，Streamlit 会缓存）。"""
    return Recommender.load(ROOT)


def main():
    st.title("🎬 智能推荐系统 Demo")
    st.caption("双塔召回（Faiss 向量检索） → DeepFM 精排 —— 工业界标准两段式架构")

    # ---- 产物自检：缺文件时给出可执行的修复指引，而不是抛栈 ----
    missing = check_artifacts(ROOT)
    if missing:
        st.error("缺少运行 Demo 所需的模型/数据文件，无法启动推荐流水线。")
        st.dataframe(pd.DataFrame(
            [{"缺失文件": rel, "说明": desc, "生成方式": cmd}
             for rel, desc, cmd in missing]),
            hide_index=True, use_container_width=True)
        st.stop()

    # ---- 加载模型/数据：损坏或版本不匹配时降级为友好报错 ----
    try:
        rec = get_recommender()
    except Exception as exc:  # 权重损坏、字段错位、依赖缺失等
        st.error(f"模型或数据加载失败：{type(exc).__name__}: {exc}")
        st.info("请确认 checkpoints/ 下的权重与当前代码版本匹配；"
                "如权重损坏，可用 src/train_two_tower.py、src/train_deepfm.py 重新训练。")
        st.stop()

    # ---------------- 侧边栏：选用户 ----------------
    user_id = st.sidebar.selectbox("选择一个用户 ID", rec.list_users(), index=0)
    profile = rec.user_profile(user_id)
    st.sidebar.markdown(f"**画像**：{'男' if profile['gender'] == 'M' else '女'} "
                        f"｜ 年龄段 {profile['age']} ｜ 职业编号 {profile['occupation']}")

    col1, col2 = st.columns(2)

    # ---------------- 左列：用户历史 ----------------
    with col1:
        st.subheader("📼 TA 的历史高分电影")
        history = rec.user_history(user_id, top=10)
        if not history:
            st.info("该用户没有 4 分及以上的历史评分。")
        else:
            st.dataframe(pd.DataFrame({
                "评分": [h["rating"] for h in history],
                "电影": [h["title"] for h in history],
                "类型": [h["genres"] for h in history]}),
                hide_index=True, use_container_width=True)

    # ---------------- 右列：推荐结果 ----------------
    with col2:
        st.subheader("🎯 模型推荐的 Top-10")
        with st.spinner("双塔召回 → DeepFM 精排中 ..."):
            result = rec.recommend(user_id, k=10, n_candidates=50)
        recs = result["recommendations"]
        # 空候选集：用户几乎看遍全库，或召回全部命中已看过 —— 优雅退出而非崩溃
        if not recs:
            st.warning("该用户已覆盖召回到的全部候选，暂无可推荐的新电影。")
            return
        st.caption(f"双塔从 {rec.n_items} 部电影中召回 {result['n_candidates']} 个候选，"
                   f"DeepFM 精排出 Top-{len(recs)}")
        st.dataframe(pd.DataFrame({
            "喜欢概率": [f"{r['score']:.1%}" for r in recs],
            "电影": [r["title"] for r in recs],
            "类型": [r["genres"] for r in recs]}),
            hide_index=True, use_container_width=True)

    st.divider()
    st.markdown("**架构说明**：双塔模型把用户和电影分别编码成 32 维向量，"
                "Faiss 内积检索完成召回（快，全库毫秒级）；DeepFM 融合用户画像、"
                "电影特征和它们的交叉信息，对每个候选精排（准，但只能处理少量候选）。"
                "这就是抖音/淘宝推荐的最小完整原型。")


if __name__ == "__main__":
    main()
