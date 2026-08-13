# -*- coding: utf-8 -*-
"""项目级常量单一来源（Single Source of Truth）。

ML-1M 的 18 个电影类型及其下标映射。训练脚本、服务端（`serving.py`）、
Bandit 等必须共用这一份常量：此前该列表在多个文件里各复制一份，
任何一处改错都会导致召回/精排的类型特征悄悄对不齐且难以排查。

用法：
    from constants import GENRES, GENRE_TO_IDX, N_GENRE
"""
from __future__ import annotations

GENRES = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
          "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
          "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

GENRE_TO_IDX = {g: i for i, g in enumerate(GENRES)}
N_GENRE = len(GENRES)


def genre_indices(genres_str: str, unknown: int = 0) -> list[int]:
    """把 'Action|Comedy' 形式的类型字符串转成下标列表。

    未知类型退回 ``unknown``（默认 0），保证输入数据里的异常类型不会崩。
    """
    return [GENRE_TO_IDX.get(g, unknown) for g in str(genres_str).split("|")]
