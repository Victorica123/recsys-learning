# -*- coding: utf-8 -*-
"""Tests for safe, manifest-verified checkpoint loading.

The service must refuse to unpickle an unregistered object and must refuse a
manifest-declared checkpoint whose bytes no longer match the release digest.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from checkpoint_io import (
    CheckpointIntegrityError,
    load_torch_checkpoint,
    sha256_file,
    verify_manifest_artifact,
)


class _UnsafePayload:
    """An arbitrary picklable object that weights_only=True must reject."""

    def __init__(self, value: int):
        self.value = value


class SafeLoaderTests(unittest.TestCase):
    def _write_artifact(self, root: Path, payload: object, name: str = "model.pt"):
        path = root / name
        torch.save(payload, path)
        return path

    def test_weights_only_payload_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self._write_artifact(
                root, {"model": {"weight": torch.ones(2, 3)}})
            loaded = load_torch_checkpoint(path, root=root)
            self.assertEqual(loaded["model"]["weight"].shape, (2, 3))

    def test_arbitrary_object_is_rejected_even_without_a_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self._write_artifact(root, {"bad": _UnsafePayload(7)})
            with self.assertRaises(CheckpointIntegrityError):
                load_torch_checkpoint(path, root=root)

    def test_manifest_size_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self._write_artifact(root, {"model": torch.ones(1)})
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "artifacts": {"model": {
                    "path": path.name, "bytes": path.stat().st_size + 1,
                    "sha256": sha256_file(path),
                }},
            }), encoding="utf-8")
            with self.assertRaisesRegex(CheckpointIntegrityError, "size mismatch"):
                load_torch_checkpoint(
                    path, root=root, manifest_path=manifest,
                    verify_hash=True)

    def test_manifest_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self._write_artifact(root, {"model": torch.ones(1)})
            manifest = root / "manifest.json"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            bad_digest = ("0" * 64 if digest != "0" * 64 else "1" * 64)
            manifest.write_text(json.dumps({
                "artifacts": {"model": {
                    "path": path.name, "bytes": path.stat().st_size,
                    "sha256": bad_digest,
                }},
            }), encoding="utf-8")
            with self.assertRaisesRegex(CheckpointIntegrityError, "SHA-256"):
                load_torch_checkpoint(
                    path, root=root, manifest_path=manifest,
                    verify_hash=True)

    def test_unknown_path_is_not_treated_as_a_manifest_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self._write_artifact(root, {"model": torch.ones(1)})
            self.assertFalse(verify_manifest_artifact(path, root=root))
            loaded = load_torch_checkpoint(path, root=root)
            self.assertIn("model", loaded)

    def test_hash_verification_is_opt_in_for_the_retrain_from_zero_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self._write_artifact(root, {"model": torch.ones(1)})
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "artifacts": {"model": {
                    "path": path.name, "bytes": path.stat().st_size,
                    "sha256": "0" * 64,
                }},
            }), encoding="utf-8")
            # 默认不校验：自训权重未登记时，开发复现流程仍可启动。
            self.assertIn("model", load_torch_checkpoint(
                path, root=root, manifest_path=manifest))
            # 显式开启后，同一份损坏声明会 fail closed。
            with self.assertRaises(CheckpointIntegrityError):
                load_torch_checkpoint(
                    path, root=root, manifest_path=manifest,
                    verify_hash=True)

    def test_missing_checkpoint_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(CheckpointIntegrityError, "does not exist"):
                load_torch_checkpoint(root / "missing.pt", root=root)


if __name__ == "__main__":
    unittest.main()
