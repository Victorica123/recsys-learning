# -*- coding: utf-8 -*-
"""实验清单工具：汇总所有实验配置和运行记录，生成统一视图。

用法:
    .venv/Scripts/python.exe scripts/experiment_manifest.py           # 打印表格
    .venv/Scripts/python.exe scripts/experiment_manifest.py --json    # JSON 输出
    .venv/Scripts/python.exe scripts/experiment_manifest.py --csv     # CSV 输出
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / "experiments"


def load_configs() -> List[Dict[str, Any]]:
    """加载所有 *_config.json 文件。"""
    configs = []
    for p in sorted(EXP_DIR.glob("*_config.json")):
        try:
            cfg = json.loads(p.read_text(encoding="utf-8"))
            cfg["_file"] = str(p.relative_to(ROOT))
            configs.append(cfg)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[warn] 跳过 {p.name}: {e}", file=sys.stderr)
    return configs


def load_all_logs() -> Dict[str, List[Dict[str, str]]]:
    """加载所有 *_log.csv 和 *_eval.csv 文件（不含 config JSON）。"""
    logs: Dict[str, List[Dict[str, str]]] = {}
    for p in sorted(EXP_DIR.glob("*.csv")):
        name = p.stem
        try:
            reader = csv.DictReader(p.open("r", encoding="utf-8", newline=""))
            rows = list(reader)
            if rows:
                logs[name] = rows
        except (csv.Error, OSError) as e:
            print(f"[warn] 无法读取 {p.name}: {e}", file=sys.stderr)
    return logs


def manifest_table() -> str:
    """生成人类可读的实验清单表格。"""
    configs = load_configs()
    logs = load_all_logs()

    lines = ["# 实验清单", f"生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}", ""]

    if not configs:
        lines.append("（尚无实验配置记录）")
        lines.append("")
        lines.append("提示：运行任何训练脚本（如 `python src/train_mf.py --epochs 1`）")
        lines.append("即可自动生成配置快照。")
        return "\n".join(lines)

    lines.append(f"## 实验配置 ({len(configs)} 条)")
    lines.append("")
    lines.append("| 脚本 | 运行名 | 时间 | 种子 | 设备 | 关键参数 |")
    lines.append("|------|--------|------|------|------|----------|")

    for cfg in configs:
        script = Path(cfg["script"]).stem
        name = cfg.get("run_name", "-")
        ts = cfg.get("timestamp", "-")
        seed = cfg.get("seed", "-")
        device = cfg.get("device", "-")
        # 提取关键参数（跳过种子、恢复等元参数）
        args = cfg.get("args", {})
        key_args = {k: v for k, v in args.items()
                    if k not in ("seed", "resume", "no_eval", "tag", "ckpt")}
        args_str = ", ".join(f"{k}={v}" for k, v in sorted(key_args.items())[:5])
        if len(key_args) > 5:
            args_str += ", ..."
        lines.append(f"| {script} | {name} | {ts} | {seed} | {device} | {args_str} |")

    lines.append("")

    if logs:
        lines.append(f"## 运行日志 ({len(logs)} 个)")
        lines.append("")
        for name, rows in sorted(logs.items()):
            # 显示最后一行的指标
            last = rows[-1]
            # 去掉时间戳列，只显示指标
            metric_cols = [c for c in last.keys()
                          if c not in ("timestamp", "run", "model",
                                       "policy", "step", "update", "epoch")]
            metrics = ", ".join(f"{c}={last[c]}" for c in metric_cols[:4])
            lines.append(f"- **{name}** ({len(rows)} 行): … {metrics}")
    else:
        lines.append("（尚无运行日志）")

    return "\n".join(lines)


def manifest_json() -> str:
    """生成 JSON 格式清单。"""
    return json.dumps({
        "configs": load_configs(),
        "logs": {k: v for k, v in load_all_logs().items()},
    }, ensure_ascii=False, indent=2)


def manifest_csv() -> str:
    """生成 CSV 格式清单（仅配置部分）。"""
    configs = load_configs()
    if not configs:
        return ""

    out = io.StringIO()
    fieldnames = ["script", "run_name", "timestamp", "seed", "device",
                  "epochs", "lr", "dim", "batch"]
    w = csv.DictWriter(out, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    for cfg in configs:
        row = {
            "script": Path(cfg["script"]).stem,
            "run_name": cfg.get("run_name", ""),
            "timestamp": cfg.get("timestamp", ""),
            "seed": cfg.get("seed", ""),
            "device": cfg.get("device", ""),
        }
        args = cfg.get("args", {})
        row.update({k: args.get(k, "") for k in ["epochs", "lr", "dim", "batch"]})
        w.writerow(row)
    return out.getvalue()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--json", action="store_true", help="JSON 格式输出")
    group.add_argument("--csv", action="store_true", help="CSV 格式输出")
    args = ap.parse_args()

    if args.json:
        print(manifest_json())
    elif args.csv:
        print(manifest_csv())
    else:
        print(manifest_table())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
