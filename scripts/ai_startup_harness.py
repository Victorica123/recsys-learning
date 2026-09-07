"""Print a compact, deterministic project handoff for a coding model.

This script intentionally uses only the Python standard library so it can run
before optional ML dependencies are imported.
"""
from __future__ import annotations

import argparse
import ast
import csv
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REQUIRED = [
    "README.md", "项目导学.md", "REPORT.md", "app.py",
    "feedback_app.py", "requirements.txt", "uv.lock",
    "artifacts/release_manifest.json",
    "data/ml-1m/ratings.dat", "data/ml-1m/users.dat",
    "data/ml-1m/movies.dat", "checkpoints/two_tower.pt",
]
KEY_SCRIPTS = [
    "train_mf.py", "train_deepfm.py", "train_two_tower.py", "run_bandit.py",
    "train_sasrec.py", "train_dqn_rec.py", "train_pg_rec.py",
]
PACKAGES = [
    "torch", "pandas", "numpy", "faiss-cpu", "streamlit", "starlette",
    "uvicorn", "gymnasium",
]


def csv_tail(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            rows = list(csv.reader(fh))
        return ",".join(rows[-1]) if rows else None
    except (OSError, UnicodeError, csv.Error):
        return None


def collect(check: bool) -> dict:
    required = {p: (ROOT / p).exists() for p in REQUIRED}
    syntax = {}
    source_paths = list((ROOT / "src").glob("*.py"))
    source_paths.extend((ROOT / "scripts").glob("*.py"))
    source_paths.extend((ROOT / "research_v2").rglob("*.py"))
    source_paths.extend((ROOT / "research_v3").rglob("*.py"))
    source_paths.extend((ROOT / "tests").glob("*.py"))
    source_paths.append(ROOT / "app.py")
    source_paths.append(ROOT / "feedback_app.py")
    source_paths.append(ROOT / "serve.py")
    for path in sorted(source_paths):
        key = str(path.relative_to(ROOT))
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            syntax[key] = "ok"
        except Exception as exc:  # syntax errors should be visible, not fatal to the harness
            syntax[key] = f"error: {type(exc).__name__}: {exc}"
    logs = {}
    for name in ("mf_log.csv", "deepfm_log.csv", "two_tower_log.csv", "dqn_v2_log.csv", "pg_eval.csv"):
        tail = csv_tail(ROOT / "experiments" / name)
        if tail is not None:
            logs[name] = tail
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "missing"
    return {
        "root": str(ROOT),
        "python": platform.python_version(),
        "required": required,
        "syntax": syntax,
        "logs": logs,
        "packages": packages,
        "python_ok": sys.version_info[:2] == (3, 12),
        "check_ok": (
            sys.version_info[:2] == (3, 12)
            and all(required.values())
            and all(v == "ok" for v in syntax.values())),
        "key_scripts": KEY_SCRIPTS,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="return non-zero when required assets or syntax are missing")
    parser.add_argument("--json", action="store_true", help="emit JSON for another agent/tool")
    parser.add_argument("--test", action="store_true", help="run the CPU-only unittest smoke suite in tests/ and return its exit code")
    parser.add_argument("--release", action="store_true",
                        help="verify uv.lock plus reference data/model hashes")
    parser.add_argument("--release-profile", default="serve",
                        choices=("serve", "research"))
    args = parser.parse_args()
    if args.test:
        import subprocess
        cmd = [sys.executable, "-m", "unittest", "discover", "-t", str(ROOT),
               "-s", str(ROOT / "tests"), "-v"]
        print(f"[recsys-learning smoke tests] {' '.join(cmd)}")
        return subprocess.run(cmd, cwd=str(ROOT)).returncode
    if args.release:
        import subprocess
        cmd = [
            sys.executable, str(ROOT / "scripts" / "verify_release.py"),
            "--profile", args.release_profile,
        ]
        print(f"[recsys-learning release check] {' '.join(cmd)}")
        return subprocess.run(cmd, cwd=str(ROOT)).returncode
    result = collect(args.check)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("[recsys-learning startup]")
        print(f"root: {result['root']}")
        print(f"python: {result['python']}")
        print(f"python_contract: {'PASS' if result['python_ok'] else 'FAIL (requires 3.12)'}")
        print(f"check: {'PASS' if result['check_ok'] else 'FAIL'}")
        missing = [p for p, ok in result["required"].items() if not ok]
        print("missing: " + (", ".join(missing) if missing else "none"))
        bad = [p for p, status in result["syntax"].items() if status != "ok"]
        print("syntax: " + (", ".join(bad) if bad else f"{len(result['syntax'])} files ok"))
        print("packages: " + ", ".join(f"{k}={v}" for k, v in result["packages"].items()))
        for name, tail in result["logs"].items():
            print(f"log_tail[{name}]: {tail[:180]}")
        print("next: read docs/AI_STARTUP_HARNESS.md; choose the smallest relevant smoke command")
    return 0 if (result["check_ok"] or not args.check) else 1


if __name__ == "__main__":
    raise SystemExit(main())
