# -*- coding: utf-8 -*-
"""可复用的推荐服务核心（与 Streamlit 解耦）。

把召回与可选精排流水线抽成一个 `Recommender` 类，供三方复用：
  - `app.py`            交互式 Demo（Streamlit）
  - `serve.py`          REST API（Starlette + uvicorn）
  - `tests/`            冒烟/集成测试

设计原则：
  - 不 import streamlit —— 服务核心与展示层解耦，可在无 UI 环境下起服务。
  - 默认只服务已经通过端到端评估的双塔 Top-K；排序器必须显式启用。
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

from checkpoint_io import load_torch_checkpoint  # noqa: E402  安全权重加载
from train_deepfm import FIELD_COLS  # noqa: E402  DeepFM 单值特征列顺序
from constants import GENRES, GENRE_TO_IDX  # noqa: E402  ML-1M 18 类型（单一来源）

# 服务端把特征行展开成了固定顺序的元组（见 `Recommender.recommend`），因此
# 训练脚本一旦调整 FIELD_COLS 的顺序或数量，线上打分就会静默错位。这里在
# import 时就断言死，把「训练-服务特征不一致」这个推荐系统最常见的事故
# 从线上错分降级成启动即失败。
EXPECTED_FIELD_COLS = ("user_id_idx", "movie_id_idx", "gender_idx",
                       "age_idx", "occupation_idx")
if tuple(FIELD_COLS) != EXPECTED_FIELD_COLS:
    raise RuntimeError(
        "train_deepfm.FIELD_COLS changed; serving builds feature rows in a "
        f"fixed order.\n  expected: {EXPECTED_FIELD_COLS}\n  actual:   "
        f"{tuple(FIELD_COLS)}\nUpdate Recommender.recommend before serving.")

# 默认上线只需要双塔权重 + ML-1M 三张原始表。DeepFM v0 已被端到端评估
# 证明为净损失，因此只在显式选择 ``ranking_policy="deepfm"`` 时才要求其产物。
BASE_ARTIFACTS = [
    ("checkpoints/two_tower.pt", "双塔召回模型", "python src/train_two_tower.py"),
    ("data/ml-1m/movies.dat", "电影表", "python src/download_data.py"),
    ("data/ml-1m/ratings.dat", "评分表", "python src/download_data.py"),
    ("data/ml-1m/users.dat", "用户表", "python src/download_data.py"),
]
RANKING_POLICIES = ("retrieval", "deepfm")


class UnknownUserError(KeyError):
    """请求了不在训练集里的 user_id。"""


# `user_history` 预缓存的每用户条数上限；超过则回落到 pandas 全表查询。
HISTORY_CACHE_LIMIT = 50


class ArtifactsMissingError(RuntimeError):
    """缺少模型权重或数据文件，无法构建服务。"""


def validate_ranking_policy(ranking_policy: str) -> str:
    """验证并返回排序策略，防止拼写错误静默回落到有害的旧路径。"""
    if ranking_policy not in RANKING_POLICIES:
        raise ValueError(
            f"unknown ranking_policy {ranking_policy!r}; "
            f"choose one of {RANKING_POLICIES}")
    return ranking_policy


def resolve_ranker_path(root: Path, deepfm_ckpt: str | Path | None) -> Path:
    path = (Path(deepfm_ckpt) if deepfm_ckpt is not None
            else Path("checkpoints/deepfm.pt"))
    return path if path.is_absolute() else root / path


def check_artifacts(root: Path = ROOT, *, ranking_policy: str = "retrieval",
                    deepfm_ckpt: str | Path | None = None):
    """返回所选服务策略缺失的产物；默认策略不依赖 DeepFM。"""
    ranking_policy = validate_ranking_policy(ranking_policy)
    missing = [(rel, desc, cmd) for rel, desc, cmd in BASE_ARTIFACTS
               if not (root / rel).exists()]
    if ranking_policy == "deepfm":
        ranker_path = resolve_ranker_path(root, deepfm_ckpt)
        if not ranker_path.exists():
            try:
                display = str(ranker_path.relative_to(root))
            except ValueError:
                display = str(ranker_path)
            missing.append((display, "实验用 DeepFM/残差精排模型",
                            "python src/train_deepfm.py"))
    return missing


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
    """默认双塔 Top-K、可显式启用 DeepFM 实验精排的推荐器。"""

    def __init__(self, tt, deepfm, index, maps, dfm_maps, u_attr,
                 movies, ratings, users, train, *, ranking_policy="retrieval",
                 ranker_arch="none"):
        self.ranking_policy = validate_ranking_policy(ranking_policy)
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
        self.ranker_arch = ranker_arch
        self.u_map = maps["u_map"]
        self.i_map = maps["i_map"]
        self.idx2movie = {v: k for k, v in self.i_map.items()}
        self.n_items = len(self.i_map)

        # ------------------------------------------------------------------
        # 热路径预展开：把 /recommend 里会用到的 pandas 查表全部换成纯 dict。
        # 实测（cProfile，100 次 /recommend）：`movies.loc[mid, ...]` 这类标量
        # 索引占了 39.8% 的时间，而 DeepFM 前向只占 7.9% —— 推理不是瓶颈，
        # 数据访问层才是。下面这些结构一次性构建，之后每请求零 pandas 调用。
        # ------------------------------------------------------------------
        self._title = {int(k): v for k, v in movies["title"].items()}
        self._genres = {int(k): v for k, v in movies["genres"].items()}
        # 类型字符串按 "|" 切分 + 映射成下标也一并预算好，省掉每请求 50 次 split。
        self._genre_idx = ({
            mid: [GENRE_TO_IDX.get(g, 0) for g in genres.split("|")]
            for mid, genres in self._genres.items()
        } if self.ranking_policy == "deepfm" else {})
        # 每用户已看过的物品下标：原实现每请求扫一遍 57 万行 DataFrame。
        self._seen_by_user = {
            int(u): set(int(i) for i in group)
            for u, group in train.groupby("u")["i"]
        }
        # DeepFM 的用户侧特征（性别/年龄/职业下标）也预先算好，
        # 推荐时只需一次 dict 查表，不再走 `users.loc[user_id]`。
        self._user_fields = {}
        if self.ranking_policy == "deepfm":
            for user_id in self.u_map:
                info = users.loc[user_id]
                self._user_fields[int(user_id)] = (
                    dfm_maps["user_id"][user_id],
                    dfm_maps["gender"][info.gender],
                    dfm_maps["age"][info.age],
                    dfm_maps["occupation"][info.occupation],
                )
        self._sorted_users = sorted(int(u) for u in self.u_map)
        # `torch.as_tensor` 对同一份 numpy 是零拷贝，但每请求重建 wrapper 仍有
        # 开销；用户属性表全程只读，直接缓存成张量。
        self._u_attr_t = torch.as_tensor(u_attr)
        # 历史高分电影：原实现每请求在 100 万行上做布尔掩码（实测 1.63 ms）。
        # 这里一次性按用户分组并排好序，缓存前 HISTORY_CACHE_LIMIT 条；
        # 请求更多时回落到原来的 pandas 路径，保证行为完全一致。
        self._history = {}
        liked = ratings[ratings.rating >= 4].sort_values(
            ["rating", "timestamp"], ascending=False)
        for user_id, group in liked.groupby("user_id", sort=False):
            self._history[int(user_id)] = [
                (int(mid), float(rating))
                for mid, rating in zip(
                    group["movie_id"].values[:HISTORY_CACHE_LIMIT],
                    group["rating"].values[:HISTORY_CACHE_LIMIT])
                if int(mid) in self._title
            ]
        # 交叉特征 ranker（v3 CrossDeepFM）需要两维稠密输入。物品类型矩阵和
        # 用户类型偏好在这里一次性算好；召回相似度则来自每次请求的 Faiss 打分。
        self._item_genres = None
        self._genre_affinity = None
        if (self.ranking_policy == "deepfm"
                and self.ranker_arch == "cross_deepfm"):
            self._build_cross_features()

    def _build_cross_features(self):
        """预计算 v3 交叉特征所需的物品类型矩阵与用户类型偏好。

        必须与 `src/train_reranker.py` 的构造方式逐字一致，否则训练与服务
        的特征口径会错位——这正是本项目端到端评估揭示的那类问题。
        """
        n_genres = len(GENRES)
        item_genres = np.zeros((self.n_items, n_genres), dtype=np.float32)
        for movie_id, i in self.i_map.items():
            genres = self._genre_idx.get(movie_id)
            if genres:
                for genre_index in genres:
                    item_genres[i, genre_index] = 1.0
        item_genres /= np.maximum(item_genres.sum(axis=1, keepdims=True), 1e-6)

        affinity = np.zeros((len(self.u_attr), n_genres), dtype=np.float32)
        for u, i in zip(self.train["u"].values, self.train["i"].values):
            affinity[u] += item_genres[i]
        affinity /= np.maximum(affinity.sum(axis=1, keepdims=True), 1e-6)
        self._item_genres = item_genres
        self._genre_affinity = affinity

    # ---------------------------------------------------------------- 构建
    @classmethod
    def load(cls, root: Path = ROOT,
             deepfm_ckpt: str | Path | None = None, *,
             ranking_policy: str = "retrieval",
             verify_hashes: bool | None = None) -> "Recommender":
        """加载数据与所选模型，构建 Faiss 内积索引。

        默认 ``ranking_policy="retrieval"``，不会加载或执行已证实为净损失的
        DeepFM v0。只有显式选择 ``deepfm`` 才会加载排序器。

        `deepfm_ckpt` 可指向替代精排权重（默认 `checkpoints/deepfm.pt`）。
        重训 ranker 时靠它让新旧版本走**完全相同**的召回与评估路径，
        差异只来自权重本身。
        """
        ranking_policy = validate_ranking_policy(ranking_policy)
        if deepfm_ckpt is not None and ranking_policy != "deepfm":
            raise ValueError(
                "deepfm_ckpt requires ranking_policy='deepfm'; "
                "the default retrieval policy never loads a ranker")
        missing = check_artifacts(
            root, ranking_policy=ranking_policy, deepfm_ckpt=deepfm_ckpt)
        if missing:
            names = ", ".join(rel for rel, _, _ in missing)
            raise ArtifactsMissingError(f"缺少必要产物：{names}")

        import faiss
        from train_two_tower import TwoTower, load_data as tt_load

        # 双塔：模型 + 评估时同一份预处理
        train, _test, u_attr, gid, gmask, maps = tt_load()
        # 权重从 release manifest 校验 SHA-256 后以 weights_only=True 加载；
        # map_location="cpu" 同时兼容 CUDA 保存、CPU 服务的 checkpoint。
        ckpt = load_torch_checkpoint(root / "checkpoints" / "two_tower.pt",
                                     verify_hash=verify_hashes)
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

        deepfm = dfm_maps = None
        arch = "none"
        if ranking_policy == "deepfm":
            from train_deepfm import DeepFM, load_data as dfm_load

            # DeepFM：重建字段映射后加载权重
            _, _, _, dfm_maps = dfm_load()
            field_sizes = [len(dfm_maps[c]) for c in
                           ["user_id", "movie_id", "gender", "age",
                            "occupation"]]
            ranker_path = resolve_ranker_path(root, deepfm_ckpt)
            payload = load_torch_checkpoint(ranker_path,
                                           verify_hash=verify_hashes)
            # `arch` 由 `src/train_reranker.py` 写入；原版 deepfm.pt 没有这个字段，
            # 缺省即为纯 DeepFM，保证老权重仍可作为显式失败对照加载。
            arch = payload.get("arch", "deepfm")
            if arch == "cross_deepfm":
                from train_reranker import CrossDeepFM
                deepfm = CrossDeepFM(
                    field_sizes,
                    residual=bool(payload.get("residual", False)))
            else:
                deepfm = DeepFM(field_sizes)
            deepfm.load_state_dict(payload["model"])
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
                   movies, ratings, users, train,
                   ranking_policy=ranking_policy, ranker_arch=arch)

    # ---------------------------------------------------------------- 查询
    def list_users(self):
        return list(self._sorted_users)

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
        if top <= HISTORY_CACHE_LIMIT:
            return [
                {"movie_id": mid,
                 "title": self._title[mid],
                 "genres": self._genres[mid],
                 "rating": rating}
                for mid, rating in self._history.get(int(user_id), ())[:top]
            ]
        # 罕见的大 top 请求：回落到全表查询，保持与缓存路径同样的排序口径。
        hist = self.ratings[(self.ratings.user_id == user_id)
                            & (self.ratings.rating >= 4)]
        hist = hist.sort_values(["rating", "timestamp"],
                               ascending=False).head(top)
        return [
            {"movie_id": int(row.movie_id),
             "title": self._title[int(row.movie_id)],
             "genres": self._genres[int(row.movie_id)],
             "rating": float(row.rating)}
            for _, row in hist.iterrows()
            if int(row.movie_id) in self._title
        ]

    # ---------------------------------------------------------------- 推荐
    def recommend(self, user_id: int, k: int = 10,
                  n_candidates: int = 50,
                  return_candidates: bool = False) -> dict:
        """双塔召回候选；默认直接取 Top-k，显式配置时才运行精排。

        热路径不碰 pandas：物品标题/类型、用户特征、已看集合都在 `__init__`
        里预展开成 dict（见那里的注释与 profile 数据）。

        `return_candidates=True` 时额外返回 `recall_order`（双塔召回顺序的
        movie_id 列表）和 `ranked_order`（所选策略的最终顺序）。默认策略下
        两者必须完全一致。
        离线的召回/精排联合评估靠它复用**线上同一条路径**，而不是另写一份
        评估逻辑——这正是「评估-服务不一致」最常见的来源。
        """
        if user_id not in self.u_map:
            raise UnknownUserError(user_id)
        u = self.u_map[user_id]
        seen = self._seen_by_user.get(u, frozenset())

        # 1) 双塔 + Faiss 内积检索：召回时多取 len(seen) 个再剔除已看过。
        #    k 必须钳在库存量以内 —— Faiss 在 k > ntotal 时用 -1 填充结果，
        #    而 `-1 not in seen` 会让这个哨兵值混进候选并在下游 KeyError。
        with torch.no_grad():
            uv = self.tt.user_vec(torch.tensor([u]), self._u_attr_t).numpy()
        search_k = min(n_candidates + len(seen), self.n_items)
        similarity, cand = self.index.search(uv.astype(np.float32), search_k)
        # 保留召回相似度：v3 的 ranker 把它当特征用。原实现直接丢掉了这个分数，
        # 精排因此完全看不到召回阶段已经算出的相关性信号。
        kept = [(int(i), float(s))
                for i, s in zip(cand[0], similarity[0])
                if i >= 0 and i not in seen][:n_candidates]
        cand = [i for i, _ in kept]
        tt_scores = {i: s for i, s in kept}
        if not cand:
            empty = {"user_id": int(user_id), "n_candidates": 0,
                     "ranking_policy": self.ranking_policy,
                     "score_type": ("two_tower_similarity"
                                    if self.ranking_policy == "retrieval"
                                    else "positive_rating_probability"),
                     "recommendations": []}
            if return_candidates:
                empty |= {"recall_order": [], "ranked_order": []}
            return empty

        # 2) 把 Faiss 下标转换成可展示的电影。默认服务策略到这里就结束，
        #    分数是归一化双塔向量的内积（余弦相似度），不是点击/喜欢概率。
        valid = [(i, self.idx2movie[i], tt_scores[i]) for i in cand
                 if i in self.idx2movie and self.idx2movie[i] in self._title]
        if self.ranking_policy == "retrieval":
            top = valid[:k]
            recs = [Recommendation(
                rank=r + 1, movie_id=mid, title=self._title[mid],
                genres=self._genres[mid], score=float(score))
                for r, (_i, mid, score) in enumerate(top)]
            recall_order = [mid for _i, mid, _score in valid]
            result = {
                "user_id": int(user_id),
                "n_candidates": len(valid),
                "ranking_policy": self.ranking_policy,
                "score_type": "two_tower_similarity",
                "recommendations": [asdict(r) for r in recs],
            }
            if return_candidates:
                result |= {"recall_order": recall_order,
                           "ranked_order": list(recall_order)}
            return result

        # 3) 组装 DeepFM 特征行（跳过映射外物品，类型下标已在加载时预算好）。
        #    元组顺序 = FIELD_COLS，由模块顶部的断言守住。
        uid_idx, gender_idx, age_idx, occ_idx = self._user_fields[int(user_id)]
        movie_ids, rows, genre_lists, item_idxs = [], [], [], []
        for i, mid, _score in valid:
            genres = self._genre_idx.get(mid)
            if genres is None:          # 不在 movies.dat 里的物品，直接跳过
                continue
            movie_ids.append(mid)
            genre_lists.append(genres)
            item_idxs.append(i)
            rows.append((uid_idx, self.dfm_maps["movie_id"][mid],
                         gender_idx, age_idx, occ_idx))
        if not rows:
            empty = {"user_id": int(user_id), "n_candidates": len(cand),
                     "ranking_policy": self.ranking_policy,
                     "score_type": "positive_rating_probability",
                     "recommendations": []}
            if return_candidates:
                empty |= {"recall_order": [], "ranked_order": []}
            return empty

        # 4) 实验精排逐候选打"正向评分概率"，取 Top-k
        x = torch.tensor(rows, dtype=torch.long)
        gid_b, mask_b = pad_genre_batch(genre_lists)
        with torch.no_grad():
            if self.ranker_arch == "cross_deepfm":
                item_index = np.asarray(item_idxs)
                dense = torch.from_numpy(np.stack([
                    np.array([tt_scores[i] for i in item_idxs],
                             dtype=np.float32),
                    np.einsum("j,ij->i", self._genre_affinity[u],
                              self._item_genres[item_index]).astype(np.float32),
                ], axis=1))
                probs = torch.sigmoid(
                    self.deepfm(x, gid_b, mask_b, dense)).numpy()
            else:
                probs = torch.sigmoid(self.deepfm(x, gid_b, mask_b)).numpy()
        ranked = np.argsort(-probs)
        top = ranked[:k]
        recs = [Recommendation(
            rank=r + 1,
            movie_id=movie_ids[t],
            title=self._title[movie_ids[t]],
            genres=self._genres[movie_ids[t]],
            score=float(probs[t])) for r, t in enumerate(top)]
        result = {"user_id": int(user_id), "n_candidates": len(cand),
                  "ranking_policy": self.ranking_policy,
                  "score_type": "positive_rating_probability",
                  "recommendations": [asdict(r) for r in recs]}
        if return_candidates:
            # movie_ids 本身就是双塔召回的顺序（cand 未重排）。
            result |= {
                "recall_order": list(movie_ids),
                "ranked_order": [movie_ids[t] for t in ranked],
            }
        return result


if __name__ == "__main__":  # 命令行快速自测：python src/serving.py [user_id] [k]
    import json
    uid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    kk = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    rec = Recommender.load()
    print(json.dumps(rec.recommend(uid, kk), ensure_ascii=False, indent=2))
