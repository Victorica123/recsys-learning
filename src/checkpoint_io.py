# -*- coding: utf-8 -*-
"""Safe, verifiable loading of project-owned PyTorch checkpoints.

The recommendation service loads model weights at startup.  Two properties
protect that boundary:

1. ``torch.load`` always runs with ``weights_only=True``.  Project checkpoints
   contain only tensors and primitive config values; disabling the pickle
   allow-list would permit arbitrary code execution from a malicious
   ``*.pt`` file.
2. When verification is explicitly enabled (``RECSYS_VERIFY_CHECKPOINTS=1`` or
   ``serve.py --verify-checkpoint-hashes``), a checkpoint declared in
   ``artifacts/release_manifest.json`` must match its recorded size and
   SHA-256 before ``torch.load`` is called.  This turns a corrupt or tampered
   deployed artifact into a clean startup failure instead of undefined runtime
   behaviour.  The switch defaults off so the documented retrain-from-zero
   workflow, which writes fresh checkpoints into the same paths, still starts.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "artifacts" / "release_manifest.json"
HASH_CHUNK_SIZE = 1024 * 1024


class CheckpointIntegrityError(RuntimeError):
    """Raised when a checkpoint cannot be loaded safely or fails verification."""


def sha256_file(path: str | Path, chunk_size: int = HASH_CHUNK_SIZE) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_release_manifest(
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> dict[str, Any] | None:
    """Return the parsed release manifest, or ``None`` in a clean checkout."""
    path = Path(manifest_path)
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CheckpointIntegrityError(
            f"release manifest {path} must contain a JSON object")
    return value


def verify_manifest_artifact(
    path: str | Path,
    *,
    root: Path = ROOT,
    manifest: dict[str, Any] | None = None,
) -> bool:
    """Return True after verifying a manifest-declared artifact.

    Unknown paths return False and are not a verification failure; they are
    optional research checkpoints that have not been pinned yet.
    """
    checkpoint = Path(path)
    if not checkpoint.is_file():
        return False
    manifest = load_release_manifest() if manifest is None else manifest
    if manifest is None:
        return False
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise CheckpointIntegrityError("release manifest 'artifacts' must be an object")
    try:
        relative = checkpoint.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        relative = None
    for artifact in artifacts.values():
        if not isinstance(artifact, dict) or artifact.get("path") != relative:
            continue
        expected_size = artifact.get("bytes")
        expected_hash = artifact.get("sha256")
        if not isinstance(expected_size, int) or not isinstance(expected_hash, str):
            raise CheckpointIntegrityError(
                f"release manifest artifact {relative!r} lacks bytes/sha256")
        actual_size = checkpoint.stat().st_size
        if actual_size != expected_size:
            raise CheckpointIntegrityError(
                f"checkpoint size mismatch for {relative}: "
                f"expected {expected_size}, got {actual_size}")
        actual_hash = sha256_file(checkpoint)
        if actual_hash != expected_hash:
            raise CheckpointIntegrityError(
                f"checkpoint SHA-256 mismatch for {relative}: "
                f"expected {expected_hash}, got {actual_hash}")
        return True
    return False


def load_torch_checkpoint(
    path: str | Path,
    *,
    map_location: Any = "cpu",
    weights_only: bool = True,
    verify_hash: bool | None = None,
    root: Path = ROOT,
    manifest_path: str | Path | None = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    """Load a project checkpoint with integrity verification and safe weights.

    ``weights_only`` cannot be disabled through this helper: project-owned
    checkpoints are required to stay inside torch's tensor/primitive allow-list.

    ``verify_hash`` is opt-in by default because the documented from-zero flow
    intentionally retrains into the pinned checkpoint paths.  Set the
    ``RECSYS_VERIFY_CHECKPOINTS=1`` environment variable (or pass True) in a
    deployed service to fail closed on any size/SHA-256 mismatch.
    """
    if verify_hash is None:
        verify_hash = os.environ.get("RECSYS_VERIFY_CHECKPOINTS") == "1"
    checkpoint = Path(path)
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    if not checkpoint.is_file():
        raise CheckpointIntegrityError(f"checkpoint does not exist: {checkpoint}")

    manifest = (
        load_release_manifest(manifest_path) if verify_hash and manifest_path
        else None)
    verify_manifest_artifact(checkpoint, root=root, manifest=manifest)

    if not weights_only:
        raise CheckpointIntegrityError(
            "unsafe checkpoint loading (weights_only=True) is disabled")
    try:
        return torch.load(checkpoint, map_location=map_location,
                          weights_only=True)
    except CheckpointIntegrityError:
        raise
    except Exception as exc:
        raise CheckpointIntegrityError(
            f"failed to load checkpoint safely: {checkpoint}: {exc}") from exc
