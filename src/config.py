# -*- coding: utf-8 -*-
"""统一配置管理：种子控制、实验参数捕获、运行清单。

所有训练脚本共享同一个 set_seed() 和 ExperimentConfig，
保证每次实验的配置可追溯、可复现。
"""
import json
import os
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

# 北京时间
_UTC8 = timezone(timedelta(hours=8))


def _git_hash(root: Path) -> str:
    """获取当前 Git commit 短哈希，失败返回 'unknown'。"""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            cwd=str(root), capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def set_seed(seed: int) -> None:
    """统一设置 Python、NumPy、PyTorch 随机种子。

    同时设置 CUDA 确定性（性能有轻微下降，用于精确复现）。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _make_run_name(script: str, args: Dict[str, Any]) -> str:
    """从脚本名和超参自动生成实验标签。"""
    base = Path(script).stem.replace("train_", "")
    parts = [f"{k}{v}" for k, v in sorted(args.items())
             if k not in ("seed", "tag", "resume", "no_eval",
                          "ckpt", "eval_every", "eval_users",
                          "epochs", "steps", "updates", "episodes",
                          "batch")]
    # 只取最后几个关键超参避免标签过长
    label = "-".join(parts[:5])
    return f"{base}_{label}" if label else base


@dataclass
class ExperimentConfig:
    """一次实验运行的完整配置快照。"""

    script: str                          # __file__ 或脚本相对路径
    timestamp: str = field(default_factory=lambda: datetime.now(_UTC8).strftime("%Y-%m-%d %H:%M:%S"))
    seed: int = 42
    device: str = "cpu"
    git_hash: str = ""
    args: Dict[str, Any] = field(default_factory=dict)
    run_name: str = ""

    def __post_init__(self):
        if not self.git_hash:
            # 自动检测项目根目录的 git hash
            root = (Path(__file__).resolve().parent.parent
                    if "__file__" in dir() else Path.cwd())
            self.git_hash = _git_hash(root)
        if not self.run_name and self.args:
            self.run_name = _make_run_name(self.script, self.args)

    def to_json(self) -> str:
        """序列化为 JSON 字符串（紧凑）。"""
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    def save(self, dir_path: Path) -> Path:
        """保存配置到指定目录，文件名为 {run_name}_config.json。"""
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        name = f"{self.run_name}_config.json" if self.run_name else "config.json"
        path = dir_path / name
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def from_args(cls, script: str, args: Any, seed: int = 42,
                  device: Optional[str] = None) -> "ExperimentConfig":
        """从 argparse.Namespace 快速创建配置快照。

        用法:
            args = ap.parse_args()
            cfg = ExperimentConfig.from_args(__file__, args, seed=42)
            cfg.save(ROOT / "experiments")
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        args_dict = {k: v for k, v in sorted(vars(args).items())}
        return cls(
            script=script,
            seed=seed,
            device=device,
            args=args_dict,
        )
