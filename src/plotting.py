# -*- coding: utf-8 -*-
"""项目内置绘图样式，避免依赖特定桌面运行时的私有 helper。"""
from __future__ import annotations

import matplotlib.pyplot as plt


def setup_plot() -> None:
    """设置可移植的中文字体回退与统一的轻量绘图样式。"""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.sans-serif": [
            "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 120,
        "savefig.dpi": 200,
    })
