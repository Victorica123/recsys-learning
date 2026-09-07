from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.verify_release import verify_manifest


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class ReleaseManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "uv.lock").write_bytes(b"lock")
        (self.root / "serve.pt").write_bytes(b"serve")
        self.manifest = {
            "schema_version": 1,
            "release_id": "test",
            "python_minor": f"{__import__('sys').version_info.major}."
                            f"{__import__('sys').version_info.minor}",
            "environment_lock": {
                "path": "uv.lock", "sha256": digest(b"lock")},
            "profiles": {"serve": ["serve"], "research": ["serve", "optional"]},
            "artifacts": {
                "serve": {
                    "path": "serve.pt", "bytes": 5, "sha256": digest(b"serve")},
                "optional": {
                    "path": "optional.pt", "bytes": 8,
                    "sha256": digest(b"optional")},
            },
        }

    def test_serve_profile_does_not_require_unselected_research_artifact(self):
        result = verify_manifest(
            self.manifest, root=self.root, profile="serve")
        self.assertTrue(result["ok"], result["errors"])

    def test_hash_mismatch_is_a_hard_failure(self):
        (self.root / "serve.pt").write_bytes(b"wrong")
        self.manifest["artifacts"]["serve"]["bytes"] = 5
        result = verify_manifest(
            self.manifest, root=self.root, profile="serve")
        self.assertFalse(result["ok"])
        self.assertIn("artifact hash mismatch: serve.pt", result["errors"])

    def test_research_profile_requires_optional_artifact(self):
        result = verify_manifest(
            self.manifest, root=self.root, profile="research")
        self.assertFalse(result["ok"])
        self.assertIn("missing artifact: optional.pt", result["errors"])

    def test_manifest_only_validates_lock_without_private_artifacts(self):
        (self.root / "serve.pt").unlink()
        result = verify_manifest(
            self.manifest, root=self.root, profile="research",
            verify_artifact_files=False)
        self.assertTrue(result["ok"], result["errors"])

    def test_parent_traversal_is_rejected(self):
        self.manifest["artifacts"]["serve"]["path"] = "../outside.pt"
        result = verify_manifest(
            self.manifest, root=self.root, profile="serve",
            verify_artifact_files=False)
        self.assertFalse(result["ok"])
        self.assertTrue(any("repository-relative" in error
                            for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
