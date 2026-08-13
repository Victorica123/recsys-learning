"""Lifecycle tests for metrics publication and graceful worker cleanup."""
import asyncio
import os
import tempfile
import unittest
from pathlib import Path

import serve
from metrics_registry import FileMetricsRegistry
from feedback import FeedbackStore


class ServeLifespanTests(unittest.TestCase):
    def test_graceful_lifespan_removes_json_and_temporary_files(self):
        original_registry = serve._REGISTRY
        original_feedback = serve._FEEDBACK
        original_interval = serve._METRICS_INTERVAL_S
        original_preload = os.environ.pop("RECSYS_PRELOAD", None)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                serve._REGISTRY = FileMetricsRegistry(
                    directory, stale_after_s=1)
                serve._FEEDBACK = FeedbackStore(
                    directory / "feedback" / "events.sqlite3")
                serve._METRICS_INTERVAL_S = 0.01

                async def scenario():
                    async with serve._lifespan(None):
                        await asyncio.sleep(0.04)
                        self.assertTrue(list(directory.glob("worker-*.json")))
                    await asyncio.sleep(0.02)

                asyncio.run(scenario())
                self.assertEqual(
                    [path for path in directory.iterdir() if path.is_file()],
                    [])
        finally:
            serve._REGISTRY = original_registry
            serve._FEEDBACK = original_feedback
            serve._METRICS_INTERVAL_S = original_interval
            if original_preload is not None:
                os.environ["RECSYS_PRELOAD"] = original_preload


if __name__ == "__main__":
    unittest.main()
