# -*- coding: utf-8 -*-
"""Verify the locked environment declaration and reference release artifacts.

Examples:
    .venv/Scripts/python.exe scripts/verify_release.py --profile serve
    .venv/Scripts/python.exe scripts/verify_release.py --profile research
    python scripts/verify_release.py --manifest-only

The script uses only the standard library so a clean CI checkout can validate
the manifest and lock before ML dependencies or private artifacts are present.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "artifacts" / "release_manifest.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_path(root: Path, relative_path: str) -> Path:
    raw = Path(relative_path)
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"path must be repository-relative: {relative_path!r}")
    root = root.resolve()
    resolved = (root / raw).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"path escapes repository: {relative_path!r}")
    return resolved


def verify_manifest(
    manifest: dict[str, Any],
    *,
    root: Path,
    profile: str,
    verify_artifact_files: bool = True,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Return a structured verification result without printing or exiting."""
    errors: list[str] = []
    checked: list[str] = []

    if manifest.get("schema_version") != 1:
        errors.append("unsupported or missing schema_version (expected 1)")

    expected_python = str(manifest.get("python_minor", ""))
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if expected_python != actual_python:
        errors.append(
            f"python minor mismatch: expected {expected_python}, got {actual_python}")

    lock = manifest.get("environment_lock")
    if not isinstance(lock, dict):
        errors.append("environment_lock must be an object")
    else:
        lock_rel = lock.get("path")
        lock_hash = lock.get("sha256")
        if not isinstance(lock_rel, str) or not SHA256_RE.fullmatch(str(lock_hash)):
            errors.append("environment_lock requires path and lowercase sha256")
        else:
            try:
                lock_path = _safe_path(root, lock_rel)
                if not lock_path.is_file():
                    errors.append(f"missing environment lock: {lock_rel}")
                else:
                    checked.append(lock_rel)
                    if verify_hashes and sha256_file(lock_path) != lock_hash:
                        errors.append(f"environment lock hash mismatch: {lock_rel}")
            except ValueError as exc:
                errors.append(str(exc))

    profiles = manifest.get("profiles")
    artifacts = manifest.get("artifacts")
    if not isinstance(profiles, dict) or profile not in profiles:
        errors.append(f"unknown release profile: {profile!r}")
        selected: list[str] = []
    else:
        selected = profiles[profile]
        if not isinstance(selected, list) or not all(
                isinstance(name, str) for name in selected):
            errors.append(f"profile {profile!r} must be a list of artifact names")
            selected = []
    if not isinstance(artifacts, dict):
        errors.append("artifacts must be an object")
        artifacts = {}

    # Validate every declaration even in --manifest-only mode. This catches a
    # malformed optional research artifact in a clean CI checkout.
    for name, entry in artifacts.items():
        if not isinstance(entry, dict):
            errors.append(f"artifact {name!r} must be an object")
            continue
        rel = entry.get("path")
        digest = entry.get("sha256")
        size = entry.get("bytes")
        if not isinstance(rel, str):
            errors.append(f"artifact {name!r} has no path")
        else:
            try:
                _safe_path(root, rel)
            except ValueError as exc:
                errors.append(str(exc))
        if not SHA256_RE.fullmatch(str(digest)):
            errors.append(f"artifact {name!r} has invalid sha256")
        if not isinstance(size, int) or size < 0:
            errors.append(f"artifact {name!r} has invalid byte size")

    for name in selected:
        if name not in artifacts:
            errors.append(f"profile {profile!r} references unknown artifact {name!r}")
            continue
        if not verify_artifact_files:
            continue
        entry = artifacts[name]
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            continue
        try:
            path = _safe_path(root, entry["path"])
        except ValueError:
            continue
        if not path.is_file():
            errors.append(f"missing artifact: {entry['path']}")
            continue
        checked.append(entry["path"])
        actual_size = path.stat().st_size
        if actual_size != entry.get("bytes"):
            errors.append(
                f"artifact size mismatch: {entry['path']} "
                f"(expected {entry.get('bytes')}, got {actual_size})")
            continue
        if verify_hashes and sha256_file(path) != entry.get("sha256"):
            errors.append(f"artifact hash mismatch: {entry['path']}")

    return {
        "ok": not errors,
        "release_id": manifest.get("release_id"),
        "profile": profile,
        "manifest_only": not verify_artifact_files,
        "hashes_verified": verify_hashes,
        "checked": checked,
        "errors": errors,
    }


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("release manifest root must be an object")
    return value


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--profile", default="serve", choices=("serve", "research"))
    parser.add_argument("--manifest-only", action="store_true",
                        help="validate declarations and uv.lock, but allow absent data/models")
    parser.add_argument("--skip-hash", action="store_true",
                        help="check paths and sizes without hashing file contents")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        manifest = load_manifest(args.manifest)
        result = verify_manifest(
            manifest,
            root=ROOT,
            profile=args.profile,
            verify_artifact_files=not args.manifest_only,
            verify_hashes=not args.skip_hash,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        result = {
            "ok": False,
            "profile": args.profile,
            "checked": [],
            "errors": [f"cannot load manifest: {exc}"],
        }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"[release verification] profile={args.profile} "
              f"status={'PASS' if result['ok'] else 'FAIL'}")
        for path in result.get("checked", []):
            print(f"ok: {path}")
        for error in result.get("errors", []):
            print(f"error: {error}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
