# -*- coding: utf-8 -*-
"""可复用的推荐服务核心（与 Streamlit 解耦）。

把 `app.py` 里内联的"双塔召回 → DeepFM 精排"两段式流水线抽成一个
`Recommender` 类，供三方复用：
  - `app.py`            交互式 Demo（Streamlit）
  - `serve.py`          REST API（Starlette + uvicorn）
  - `tests/`            冒烟/集成测试

设计原则：
  - 不 import streamlit —— 服务核心与展示层解耦，可在无 UI 环境下起服务。
  - 加载路径与 `app.py` 原实现逐字对齐，保证线上/Demo 结果一致。
  - 产物自检、未知用户、空候选集都抛显式异常或返回空结果，绝不裸崩。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from train_deepfm import FIELD_COLS  # noqa: E402  DeepFM 单值特征列顺序
from constants import GENRES, GENRE_TO_IDX  # noqa: E402  ML-1M 18 类型（单一来源）

# 上线必备产物：两个模型权重 + ML-1M 三张原始表。缺任一都无法起服务。
REQUIRED_ARTIFACTS = [
    ("checkpoints/two_tower.pt", "双塔召回模型", "python src/train_two_tower.py"),
    ("checkpoints/deepfm.pt", "DeepFM 精排模型", "python src/train_deepfm.py"),
    ("data/ml-1m/movies.dat", "电影表", "python src/download_data.py"),
    ("data/ml-1m/ratings.dat", "评分表", "python src/download_data.py"),
    ("data/ml-1m/users.dat", "用户表", "python src/download_data.py"),
]


class UnknownUserError(KeyError):
    """请求了不在训练集里的 user_id。"""


class ArtifactsMissingError(RuntimeError):
    """缺少模型权重或数据文件，无法构建服务。"""


def check_artifacts(root: Path = ROOT):
    """返回缺失产物的 (相对路径, 说明, 生成命令) 列表；空列表表示齐备。"""
    return [(rel, desc, cmd) for rel, desc, cmd in REQUIRED_ARTIFACTS
            if not (root / rel).exists()]


def pad_genre_batch(genre_lists):
    """把不等长的类型下标列表右侧补 0，返回 (gid_b, mask_b)。

    纯函数、无模型依赖 —— DeepFM 多值类型特征的打包逻辑，单独可测。
    空 batch 时返回 (0, 1) 形状的占位张量。
    """
    n = len(genre_lists)
    max_len = max((len(g) for g in genre_lists), default=1)
    gid_b = torch.zeros(n, max_len, dtype=torch.long)
    mask_b = torch.zeros(n, max_len)
    for r, gs in enumerate(genre_lists):
        if gs:
            gid_b[r, :len(gs)] = torch.tensor(gs, dtype=torch.long)
            mask_b[r, :len(gs)] = 1.0
    return gid_b, mask_b


@dataclass
class Recommendation:
    rank: int
    movie_id: int
    title: str
    genres: str
    score: float


class Recommender:
    """双塔召回 → DeepFM 精排的两段式推荐器。"""

    def __init__(self, tt, deepfm, index, maps, dfm_maps, u_attr,
                 movies, ratings, users, train):
        self.tt = tt
        self.deepfm = deepfm
        self.index = index
        self.maps = maps
        self.dfm_maps = dfm_maps
        self.u_attr = u_attr
        self.movies = movies
        self.ratings = ratings
        self.users = users
        self.train = train
        self.u_map = maps["u_map"]
        self.i_map = maps["i_map"]
        self.idx2movie = {v: k for k, v in self.i_map.items()}
        self.n_items = len(self.i_map)

    # ---------------------------------------------------------------- 构建
    @classmethod
    def load(cls, root: Path = ROOT) -> "Recommender":
        """加载数据 + 两个模型，构建 Faiss 内积索引。缺产物时抛显式异常。"""
        missing = check_artifacts(root)
        if missing:
            names = ", ".join(rel for rel, _, _ in missing)
            raise ArtifactsMissingError(f"缺少必要产物：{names}")

        import faiss
        from train_two_tower import TwoTower, load_data as tt_load
        from train_deepfm import DeepFM, load_data as dfm_load

        # 双塔：模型 + 评估时同一份预处理
        train, _test, u_attr, gid, gmask, maps = tt_load()
        ckpt = torch.load(root / "checkpoints" / "two_tower.pt",
                          weights_only=False)
        tt = TwoTower(ckpt["n_users"], ckpt["n_items"])
        tt.load_state_dict(ckpt["model"])
        tt.eval()

        gid_t = torch.as_tensor(gid)
        gmask_t = torch.as_tensor(gmask)
        with torch.no_grad():
            item_vecs = tt.item_vec(torch.arange(ckpt["n_items"]),
                                    gid_t, gmask_t).numpy()
        index = faiss.IndexFlatIP(item_vecs.shape[1])
        index.add(item_vecs.astype(np.float32))

        # DeepFM：重建字段映射后加载权重
        _, _, _, dfm_maps = dfm_load()
        field_sizes = [len(dfm_maps[c]) for c in
                       ["user_id", "movie_id", "gender", "age", "occupation"]]
        deepfm = DeepFM(field_sizes)
        deepfm.load_state_dict(
            torch.load(root / "checkpoints" / "deepfm.pt",
                       weights_only=False)["model"])
        deepfm.eval()

        movies = pd.read_csv(root / "data" / "ml-1m" / "movies.dat", sep="::",
                             engine="python",
                             names=["movie_id", "title", "genres"],
                             encoding="latin-1").set_index("movie_id")
        ratings = pd.read_csv(root / "data" / "ml-1m" / "ratings.dat", sep="::",
                              engine="python",
                              names=["user_id", "movie_id", "rating",
                                     "timestamp"])
        users = pd.read_csv(root / "data" / "ml-1m" / "users.dat", sep="::",
                            engine="python",
                            names=["user_id", "gender", "age", "occupation",
                                   "zip"]).set_index("user_id")
        return cls(tt, deepfm, index, maps, dfm_maps, u_attr,
                   movies, ratings, users, train)

    # ---------------------------------------------------------------- 查询
    def list_users(self):
        return sorted(int(u) for u in self.u_map.keys())

    def user_profile(self, user_id: int) -> dict:
        if user_id not in self.u_map:
            raise UnknownUserError(user_id)
        info = self.users.loc[user_id]
        return {"user_id": int(user_id),
                "gender": "M" if info.gender == "M" else "F",
                "age": int(info.age),
                "occupation": int(info.occupation)}

    def user_history(self, user_id: int, top: int = 10):
        """返回用户历史高分（≥4）电影，最多 top 条。"""
        if user_id not in self.u_map:
            raise UnknownUserError(user_id)
        hist = self.ratings[(self.ratings.user_id == user_id)
                            & (self.ratings.rating >= 4)]
        hist = hist.sort_values(["rating", "timestamp"],
                                ascending=False).head(top)
        out = []
        for _, row in hist.iterrows():
            mid = int(row.movie_id)
            if mid in self.movies.index:
                out.append({"movie_id": mid,
                            "title": self.movies.loc[mid, "title"],
                            "genres": self.movies.loc[mid, "genres"],
                            "rating": float(row.rating)})
        return out

    # ---------------------------------------------------------------- 推荐
    def recommend(self, user_id: int, k: int = 10,
                  n_candidates: int = 50) -> dict:
        """双塔召回 n_candidates 个候选 → DeepFM 精排出 Top-k。"""
        if user_id not in self.u_map:
            raise UnknownUserError(user_id)
        u = self.u_map[user_id]
        seen = set(self.train[self.train.u == u]["i"].tolist())

        # 1) 双塔 + Faiss 内积检索：召回时多取 len(seen) 个再剔除已看过
        with torch.no_grad():
            uv = self.tt.user_vec(torch.tensor([u]),
                                  torch.as_tensor(self.u_attr)).numpy()
        _, cand = self.index.search(uv.astype(np.float32),
                                    n_candidates + len(seen))
        cand = [int(i) for i in cand[0] if i not in seen][:n_candidates]
        if not cand:
            return {"user_id": int(user_id), "n_candidates": 0,
                    "recommendations": []}

        # 2) 组装 DeepFM 特征行（跳过映射外物品、未知类型退回 0）
        info = self.users.loc[user_id]
        rows, genre_lists = [], []
        for i in cand:
            mid = self.idx2movie[i]
            if mid not in self.movies.index:
                continue
            gs = [GENRE_TO_IDX.get(g, 0)
                  for g in self.movies.loc[mid, "genres"].split("|")]
            genre_lists.append(gs)
            rows.append({
                "user_id_idx": self.dfm_maps["user_id"][user_id],
                "movie_id_idx": self.dfm_maps["movie_id"][mid],
                "gender_idx": self.dfm_maps["gender"][info.gender],
                "age_idx": self.dfm_maps["age"][info.age],
                "occupation_idx": self.dfm_maps["occupation"][info.occupation],
                "movie_id": mid})
        if not rows:
            return {"user_id": int(user_id), "n_candidates": len(cand),
                    "recommendations": []}

        # 3) DeepFM 逐候选打"喜欢概率"，取 Top-k
        x = torch.tensor([[r[c] for c in FIELD_COLS] for r in rows])
        gid_b, mask_b = pad_genre_batch(genre_lists)
        with torch.no_grad():
            probs = torch.sigmoid(self.deepfm(x, gid_b, mask_b)).numpy()
        top = np.argsort(-probs)[:k]
        recs = [Recommendation(
            rank=r + 1,
            movie_id=int(rows[t]["movie_id"]),
            title=self.movies.loc[rows[t]["movie_id"], "title"],
            genres=self.movies.loc[rows[t]["movie_id"], "genres"],
            score=float(probs[t])) for r, t in enumerate(top)]
        return {"user_id": int(user_id), "n_candidates": len(cand),
                "recommendations": [asdict(r) for r in recs]}


if __name__ == "__main__":  # 命令行快速自测：python src/serving.py [user_id] [k]
    import json
    uid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    kk = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    rec = Recommender.load()
    print(json.dumps(rec.recommend(uid, kk), ensure_ascii=False, indent=2))
