# -*- coding: utf-8 -*-
"""真实反馈日志 → 离线 RL 数据管线（第一步：上下文 Bandit 粒度）。

把 ``runtime/feedback/events.sqlite3`` 里的真实曝光/反馈日志导出，重建成
(state=SASRec 用户上下文, action=Top200 动作下标, reward=事件奖励) 三元组。

现实约束与诚实门槛
------------------
1. 真实日志是「每次推荐一屏」的独立曝光，不是按会话顺序记录的 MDP 序列；
   因此这里先按**上下文 Bandit**粒度建数据集，真正的 (s,a,r,s',done) 会话
   MDP 离线 RL 需要服务端按会话追加 state 转移日志（后续增强）。
2. 离线 RL 极易在小样本上自举出伪结论。数据不足时本脚本**拒绝训练**，
   只输出数据画像与阻塞项，绝不硬练一个数字出来。

用法
----
    .venv/Scripts/python.exe scripts/feedback_offline_rl.py
    .venv/Scripts/python.exe scripts/feedback_offline_rl.py --min-transitions 2000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from feedback import FeedbackStore, feedback_reward  # noqa: E402
from train_dqn_rec import load_user_model, load_rec_assets  # noqa: E402

SASREC_CKPT = ROOT / "checkpoints" / "sasrec_main.pt"
FEEDBACK_DB = ROOT / "runtime" / "feedback" / "events.sqlite3"
DATA = ROOT / "data" / "ml-1m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_movie_item_map():
    """重建 SASRec 的 movie_id → 物品下标(从 1 开始) 映射，与训练时一致。"""
    ratings = pd.read_csv(DATA / "ratings.dat", sep="::", engine="python",
                          names=["user_id", "movie_id", "rating", "timestamp"])
    return {int(m): i + 1 for i, m in enumerate(sorted(ratings["movie_id"].unique()))}


def user_context(sasrec, maxlen, i_map, movie_ids):
    """用冻结 SASRec 把用户历史 movie_ids 编码成 64 维上下文向量。"""
    items = [i_map[m] for m in movie_ids if m in i_map][-maxlen:]
    if not items:
        return None
    seq = torch.tensor([items], device=DEVICE)
    with torch.no_grad():
        ctx = sasrec.encode(seq)[0, -1].cpu().numpy()
    return ctx.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-transitions", type=int, default=2000,
                    help="训练所需最小转移数，低于则拒绝训练")
    ap.add_argument("--min-actors", type=int, default=3,
                    help="训练所需最小匿名体验者数")
    ap.add_argument("--min-actions", type=int, default=20,
                    help="训练所需最小被覆盖动作数")
    ap.add_argument("--out", default="experiments/feedback_offline_dataset.npz")
    args = ap.parse_args()

    if not FEEDBACK_DB.exists():
        print(f"[feedback-offline-rl] 找不到反馈库 {FEEDBACK_DB}，先跑服务收集数据。")
        return 0

    store = FeedbackStore(FEEDBACK_DB)
    rows = store.export_rows()  # 候选级 replay 行（含 reward）

    # 只保留「真实曝光过 + 有反馈」的样本
    impressed = [r for r in rows if r.get("impressed") and r.get("reward") is not None]
    print(f"[feedback-offline-rl] 候选 {len(rows)} / 曝光且含反馈 {len(impressed)}")

    if not impressed:
        print("[feedback-offline-rl] 无可用反馈样本，无法建离线数据集。")
        return 0

    # 用户历史（评分）按时间排序，用于重建 SASRec 上下文
    ratings = pd.read_csv(DATA / "ratings.dat", sep="::", engine="python",
                          names=["user_id", "movie_id", "rating", "timestamp"])
    hist = ratings.sort_values("timestamp").groupby("user_id")["movie_id"] \
                 .apply(list).to_dict()

    sasrec, maxlen, _ = load_user_model(SASREC_CKPT)
    top_items, item_genre, _ = load_rec_assets()
    top_set = {int(i): k for k, i in enumerate(top_items)}  # 物品下标 → 动作下标
    i_map = build_movie_item_map()

    states, actions, rewards, actors = [], [], [], []
    skipped_ood, skipped_ctx = 0, 0
    actor_set = set()
    for r in impressed:
        uid = int(r["user_id"])
        movie_id = int(r["movie_id"])
        item = i_map.get(movie_id)
        if item is None or item not in top_set:
            skipped_ood += 1  # 不在 Top200 动作空间，无法作为可学习动作
            continue
        ctx = user_context(sasrec, maxlen, i_map, hist.get(uid, []))
        if ctx is None:
            skipped_ctx += 1
            continue
        actor = str(r.get("actor_id") or "unknown")
        states.append(ctx)
        actions.append(top_set[item])
        rewards.append(float(r["reward"]))
        actors.append(actor)
        actor_set.add(actor)

    print(f"[feedback-offline-rl] 可建转移 {len(states)}"
          f"（跳过 OOD 动作 {skipped_ood}，空上下文 {skipped_ctx}）")

    blockers = []
    if len(states) < args.min_transitions:
        blockers.append(f"transitions {len(states)} < {args.min_transitions}")
    if len(actor_set) < args.min_actors:
        blockers.append(f"actors {len(actor_set)} < {args.min_actors}")
    unique_actions = len(set(actions))
    if unique_actions < args.min_actions:
        blockers.append(f"covered_actions {unique_actions} < {args.min_actions}")

    if states:
        st = np.stack(states)
        rw = np.array(rewards, dtype=np.float32)
        print(f"[feedback-offline-rl] 数据画像: 状态维度 {st.shape[1]}, "
              f"奖励均值 {rw.mean():.3f}, 覆盖动作 {unique_actions}, "
              f"体验者 {len(actor_set)}")
        np.savez_compressed(args.out, states=st, actions=np.array(actions),
                            rewards=rw, actors=np.array(actors))
        print(f"[feedback-offline-rl] 已保存 {args.out}")

    if blockers:
        print("[feedback-offline-rl] 数据不足，拒绝训练（诚实门槛）：")
        for b in blockers:
            print(f"  - {b}")
        print("[feedback-offline-rl] 结论：继续采集真实反馈后再做离线 RL；"
              "当前仅完成数据管线与门槛校验。")
        return 0

    # 数据充足时（未来）接 research_v2/phase2_offline_rl 的 naive DQN/BC 训练。
    print("[feedback-offline-rl] 数据达标，可接入离线 RL 训练（暂未在本脚本内实现）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
