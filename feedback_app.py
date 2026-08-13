# -*- coding: utf-8 -*-
"""Interactive real-human feedback collector for the recommendation project.

Run from the project root:

    .venv/Scripts/python.exe -m streamlit run feedback_app.py --server.port 8502
"""
from __future__ import annotations

import os
import sys
import uuid
from collections import defaultdict
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from feedback import (FeedbackStore, compare_logged_policies,  # noqa: E402
                      select_slate)
from serving import Recommender, check_artifacts  # noqa: E402


DEFAULT_DB = Path(os.environ.get(
    "RECSYS_FEEDBACK_DB", ROOT / "runtime" / "feedback" / "events.sqlite3"))
MODEL_VERSION = os.environ.get(
    "RECSYS_MODEL_VERSION", "two-tower-deepfm-ml1m-v1")

PILOT_MODE = "快速个人试验"
FORMAL_MODE = "多人正式评估"
PILOT_POLICY = "epsilon_slate_personal_pilot_v1"
FORMAL_POLICY = "epsilon_slate_formal_v1"

PILOT_K = 5
PILOT_POOL_SIZE = 10
PILOT_EXPLORATION_RATE = 0.20
PILOT_MIN_RECOMMENDATIONS = 20
PILOT_MIN_ESS = 5.0

FORMAL_MIN_RECOMMENDATIONS = 200
FORMAL_MIN_ACTORS = 5
FORMAL_MIN_ESS = 100.0

st.set_page_config(
    page_title="推荐反馈实验台", page_icon="🎬", layout="wide")


@st.cache_resource
def get_recommender() -> Recommender:
    return Recommender.load(ROOT)


@st.cache_resource
def get_feedback_store(path: str) -> FeedbackStore:
    store = FeedbackStore(path)
    store.initialize()
    return store


def create_slate(
    rec: Recommender,
    store: FeedbackStore,
    *,
    mode: str,
    actor_id: str,
    user_id: int,
    k: int,
    pool_size: int,
    exploration_rate: float,
) -> dict:
    result = rec.recommend(
        user_id, k=pool_size, n_candidates=max(50, pool_size))
    served, logged = select_slate(
        result["recommendations"], k, exploration_rate)
    if not logged:
        raise RuntimeError("模型没有生成可记录的候选")

    recommendation_id = uuid.uuid4().hex
    policy_name = PILOT_POLICY if mode == PILOT_MODE else FORMAL_POLICY
    store.record_recommendation(
        recommendation_id=recommendation_id,
        request_id=f"streamlit-{uuid.uuid4().hex}",
        actor_id=actor_id,
        user_id=user_id,
        model_version=MODEL_VERSION,
        policy_name=policy_name,
        exploration_rate=exploration_rate,
        requested_k=k,
        candidates=logged,
    )
    store.record_impressions(
        recommendation_id,
        [item["movie_id"] for item in served],
    )
    return {
        "signature": (
            mode, actor_id, user_id, k, pool_size, exploration_rate),
        "recommendation_id": recommendation_id,
        "recommendations": served,
    }


def record_preference(
    store: FeedbackStore,
    slate: dict,
    actor_id: str,
    movie_id: int,
    event_type: str,
) -> None:
    store.record_feedback(
        recommendation_id=slate["recommendation_id"],
        movie_id=movie_id,
        event_type=event_type,
        event_id=(
            f"ui:{actor_id}:{slate['recommendation_id']}:"
            f"{movie_id}:{event_type}"
        ),
    )
    st.session_state[
        f"preference:{slate['recommendation_id']}:{movie_id}"
    ] = event_type


def rows_for_mode(rows: list[dict], mode: str, actor_id: str) -> list[dict]:
    if mode == PILOT_MODE:
        return [
            row for row in rows
            if row.get("policy_name") == PILOT_POLICY
            and str(row.get("actor_id")) == actor_id
        ]
    return [
        row for row in rows
        if row.get("policy_name") != PILOT_POLICY
    ]


def completed_choice_rows(rows: list[dict]) -> list[dict]:
    """Keep quick-mode batches with one explicit response for every shown item."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["recommendation_id"])].append(row)
    completed_ids = {
        recommendation_id
        for recommendation_id, candidates in grouped.items()
        if (
            (served := [
                row for row in candidates
                if row.get("served_rank") is not None
            ])
            and all(row.get("feedback") for row in served)
        )
    }
    return [
        row for row in rows
        if str(row["recommendation_id"]) in completed_ids
    ]


def experiment_stats(rows: list[dict]) -> dict:
    recommendation_ids = {
        str(row["recommendation_id"]) for row in rows}
    served = sum(row.get("served_rank") is not None for row in rows)
    impressed = sum(bool(row.get("impressed")) for row in rows)
    return {
        "recommendations": len(recommendation_ids),
        "actors": len({
            str(row.get("actor_id") or "__unknown__") for row in rows}),
        "impressions": impressed,
        "feedback_events": sum(len(row.get("feedback") or []) for row in rows),
        "impression_coverage": impressed / served if served else None,
    }


def submit_quick_choice(
    store: FeedbackStore,
    slate: dict,
    actor_id: str,
    selected_movie_id: int,
) -> None:
    """Log a listwise choice: one click and explicit skips for the alternatives."""
    for item in slate["recommendations"]:
        movie_id = int(item["movie_id"])
        event_type = "click" if movie_id == selected_movie_id else "skip"
        record_preference(
            store, slate, actor_id, movie_id, event_type)


def render_quick_batch(
    store: FeedbackStore,
    slate: dict,
    actor_id: str,
    exploration_rate: float,
) -> None:
    st.subheader("这一批，你最想看哪一部？")
    st.caption(
        f"批次 `{slate['recommendation_id']}` · 探索率 "
        f"{exploration_rate:.0%}。点一部电影即完成本批并自动进入下一批。")
    for item in slate["recommendations"]:
        movie_id = int(item["movie_id"])
        label = (
            f"{item['title']} · {item['genres']} · "
            f"模型分数 {item['score']:.1%}"
        )
        if st.button(
            label,
            key=f"quick-choice:{slate['recommendation_id']}:{movie_id}",
            use_container_width=True,
        ):
            try:
                submit_quick_choice(
                    store, slate, actor_id, movie_id)
                st.session_state.pop("feedback_slate", None)
                st.rerun()
            except Exception as exc:
                st.error(f"反馈写入失败：{exc}")


def render_formal_batch(
    store: FeedbackStore,
    slate: dict,
    actor_id: str,
    exploration_rate: float,
) -> None:
    st.subheader("请按真实偏好评价这一批电影")
    st.caption(
        f"推荐批次 `{slate['recommendation_id']}` · "
        f"探索率 {exploration_rate:.0%}。未操作的曝光按 0 奖励处理。")
    for item in slate["recommendations"]:
        movie_id = int(item["movie_id"])
        response_key = (
            f"preference:{slate['recommendation_id']}:{movie_id}")
        response = st.session_state.get(response_key)
        info, want, like, dislike, skip = st.columns([6, 1, 1, 1, 1])
        with info:
            st.markdown(
                f"**{item['rank']}. {item['title']}**  \n"
                f"`{item['genres']}` · 模型分数 {item['score']:.1%}")
        actions = [
            (want, "想看", "click"),
            (like, "喜欢", "like"),
            (dislike, "不喜欢", "dislike"),
            (skip, "跳过", "skip"),
        ]
        for column, label, event_type in actions:
            with column:
                if st.button(
                    label,
                    key=f"{response_key}:{event_type}",
                    disabled=response is not None,
                    use_container_width=True,
                ):
                    try:
                        record_preference(
                            store, slate, actor_id, movie_id, event_type)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"反馈写入失败：{exc}")
        if response:
            st.success(f"已记录：{response}")
        st.divider()


def decision_text(decision: str, pilot: bool) -> str:
    if pilot:
        return {
            "promote_candidate": "个人试验倾向：采用多样性重排",
            "keep_baseline": "个人试验倾向：保留 DeepFM 基线",
            "continue_experiment": "个人试验暂不明确，建议继续积累",
            "collect_more_data": "个人试验数据质量暂未过门槛",
        }[decision]
    return {
        "promote_candidate": "正式结论：采用多样性重排",
        "keep_baseline": "正式结论：保留 DeepFM 基线",
        "continue_experiment": "正式结论：当前差异不显著，继续实验",
        "collect_more_data": "尚未达到正式评估门槛",
    }[decision]


def main() -> None:
    st.title("🎬 推荐反馈实验台")
    st.caption(
        "记录真人对推荐结果的曝光与偏好；不会把 MovieLens 历史评分或"
        "合成点击冒充线上反馈。")

    missing = check_artifacts(ROOT)
    if missing:
        st.error("缺少模型或数据产物，无法启动实验台。")
        st.code("\n".join(f"{path}: {command}" for path, _, command in missing))
        st.stop()
    try:
        rec = get_recommender()
        store = get_feedback_store(str(DEFAULT_DB))
    except Exception as exc:
        st.error(f"初始化失败：{type(exc).__name__}: {exc}")
        st.stop()

    mode = st.sidebar.selectbox(
        "实验模式", [PILOT_MODE, FORMAL_MODE], index=0)
    actor_id = st.sidebar.text_input(
        "体验者代号",
        value=st.session_state.get("actor_id", "local-evaluator"),
        help="仅用于区分评价者；请勿填写姓名、邮箱等个人信息。",
    ).strip()
    if not actor_id:
        st.warning("请先填写一个非空匿名体验者代号。")
        st.stop()
    if len(actor_id) > 128:
        st.warning("体验者代号不能超过 128 个字符。")
        st.stop()
    st.session_state["actor_id"] = actor_id

    user_id = st.sidebar.selectbox(
        "选择 MovieLens 用户画像", rec.list_users(), index=0)
    if mode == PILOT_MODE:
        k = PILOT_K
        pool_size = PILOT_POOL_SIZE
        exploration_rate = PILOT_EXPLORATION_RATE
        min_recommendations = PILOT_MIN_RECOMMENDATIONS
        min_actors = 1
        min_ess = PILOT_MIN_ESS
        confidence = 0.90
        bootstrap_samples = 1000
        st.sidebar.info(
            "每批展示 5 部，点 1 部即自动进入下一批。完成 20 批后给出个人试验信号；"
            "它不能替代多人正式上线结论。")
    else:
        k = st.sidebar.slider("每批展示数量", 5, 10, 10)
        pool_size = st.sidebar.slider(
            "探索候选池", k, 30, max(20, k))
        exploration_rate = st.sidebar.slider(
            "均匀探索率", 0.05, 0.20, 0.10, 0.01,
            help="10% 表示每批有 10% 概率从候选池均匀抽取。")
        min_recommendations = FORMAL_MIN_RECOMMENDATIONS
        min_actors = FORMAL_MIN_ACTORS
        min_ess = FORMAL_MIN_ESS
        confidence = 0.95
        bootstrap_samples = 2000
        st.sidebar.info(
            "正式门槛：至少 200 批、5 位匿名体验者、95% 曝光覆盖，"
            "且有效样本量 ESS ≥ 100。")

    all_rows = store.export_rows()
    selected_rows = rows_for_mode(all_rows, mode, actor_id)
    analysis_rows = (
        completed_choice_rows(selected_rows)
        if mode == PILOT_MODE else selected_rows
    )
    stats = experiment_stats(analysis_rows)
    pilot_complete = (
        mode == PILOT_MODE
        and stats["recommendations"] >= min_recommendations
    )

    signature = (
        mode, actor_id, user_id, k, pool_size, exploration_rate)
    slate = st.session_state.get("feedback_slate")
    if slate is not None and slate.get("signature") != signature:
        st.session_state.pop("feedback_slate", None)
        slate = None

    if mode == FORMAL_MODE and st.button(
        "🔄 换一批并记录新曝光", type="primary"
    ):
        st.session_state.pop("feedback_slate", None)
        slate = None

    if not pilot_complete and slate is None:
        try:
            with st.spinner("生成推荐并记录曝光…"):
                slate = create_slate(
                    rec,
                    store,
                    mode=mode,
                    actor_id=actor_id,
                    user_id=user_id,
                    k=k,
                    pool_size=pool_size,
                    exploration_rate=exploration_rate,
                )
            st.session_state["feedback_slate"] = slate
        except Exception as exc:
            st.error(f"生成或记录推荐失败：{type(exc).__name__}: {exc}")
            st.stop()

    if pilot_complete:
        st.success("20 批个人快速试验已完成，不需要再继续点了。")
        st.session_state.pop("feedback_slate", None)
    elif mode == PILOT_MODE:
        render_quick_batch(store, slate, actor_id, exploration_rate)
    else:
        render_formal_batch(store, slate, actor_id, exploration_rate)

    st.subheader("采集进度")
    metric_columns = st.columns(4)
    metric_columns[0].metric("完整批次", stats["recommendations"])
    metric_columns[1].metric("已曝光物品", stats["impressions"])
    metric_columns[2].metric("反馈事件", stats["feedback_events"])
    metric_columns[3].metric(
        "匿名体验者", stats["actors"])
    progress = min(
        1.0, stats["recommendations"] / min_recommendations)
    label = "个人试验" if mode == PILOT_MODE else "正式评估"
    st.progress(
        progress,
        text=(
            f"{label}样本门槛：{stats['recommendations']} / "
            f"{min_recommendations} 批"
        ),
    )

    if analysis_rows:
        comparison = compare_logged_policies(
            analysis_rows,
            target_k=k,
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
            min_recommendations=min_recommendations,
            min_actors=min_actors,
            min_effective_sample_size=min_ess,
            min_impression_coverage=1.0 if mode == PILOT_MODE else 0.95,
        )
        gate = comparison["gate"]
        message = decision_text(
            gate["decision"], mode == PILOT_MODE)
        if gate["formal_ready"]:
            st.success(message)
        else:
            st.warning(
                message + "：" + "；".join(gate["blockers"]))
        expander_label = (
            "查看个人试验估计（仅用于选择下一步）"
            if mode == PILOT_MODE
            else "查看正式策略评估"
        )
        with st.expander(expander_label):
            st.json(comparison)


if __name__ == "__main__":
    main()
