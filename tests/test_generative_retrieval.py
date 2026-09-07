import unittest

import numpy as np
import pandas as pd
import torch

from src.generative_retrieval import (
    SemanticIDGenerator,
    SemanticIDIndex,
    SemanticSequenceDataset,
    append_collision_tokens,
    residual_kmeans,
    semantic_recommendations,
)


class SemanticIDTests(unittest.TestCase):
    def test_residual_quantization_is_deterministic_and_reduces_error(self):
        rng = np.random.default_rng(7)
        vectors = rng.normal(size=(40, 8)).astype(np.float32)
        codes1, _, mse1 = residual_kmeans(vectors, (4, 4), seed=9, iterations=8)
        codes2, _, mse2 = residual_kmeans(vectors, (4, 4), seed=9, iterations=8)
        np.testing.assert_array_equal(codes1, codes2)
        self.assertEqual(mse1, mse2)
        self.assertLessEqual(mse1[1], mse1[0])

    def test_collision_token_makes_every_item_id_unique(self):
        base = np.asarray([[0, 1], [0, 1], [0, 2], [0, 1]], dtype=np.int64)
        codes, cardinality = append_collision_tokens(base)
        self.assertEqual(cardinality, 3)
        self.assertEqual(len({tuple(row) for row in codes.tolist()}), 4)
        self.assertEqual(codes[:, -1].tolist(), [0, 1, 0, 2])

    def test_index_profile_reports_every_final_id(self):
        vectors = np.eye(12, dtype=np.float32)
        index = SemanticIDIndex.fit(vectors, (3, 3), seed=3, iterations=5)
        profile = index.profile()
        self.assertEqual(profile["items"], 12)
        self.assertEqual(profile["unique_final_ids"], 12)
        self.assertEqual(profile["levels"], 3)


class AutoregressiveGeneratorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.model = SemanticIDGenerator(
            (3, 3, 2), max_history_items=2, d_model=12,
            n_heads=3, n_layers=1, dropout=0.0)
        self.model.eval()

    def test_future_target_tokens_do_not_change_earlier_logits(self):
        context = torch.tensor([[[0, 1, 0], [1, 2, 1]]])
        target1 = torch.tensor([[1, 0, 0]])
        target2 = torch.tensor([[1, 2, 1]])
        logits1 = self.model(context, target1)
        logits2 = self.model(context, target2)
        torch.testing.assert_close(logits1[0], logits2[0])
        torch.testing.assert_close(logits1[1], logits2[1])

    def test_full_catalog_recommendation_excludes_complete_history(self):
        codes = np.asarray([
            [0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [2, 0, 0],
        ], dtype=np.int64)
        train = pd.DataFrame({
            "u": [0, 0], "i": [0, 1], "timestamp": [1, 2],
        })
        recs = semantic_recommendations(
            self.model, train, [0], codes, k=2,
            max_history_items=2, user_batch=1, prefix_chunk=2)
        self.assertEqual(len(recs[0]), 2)
        self.assertFalse(set(recs[0]) & {0, 1})

    def test_temporal_validation_dataset_uses_only_later_targets(self):
        codes = np.asarray([
            [0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0],
        ], dtype=np.int64)
        frame = pd.DataFrame({
            "u": [0, 0, 0, 0], "i": [0, 1, 2, 3],
            "timestamp": [1, 2, 3, 4],
        })
        dataset = SemanticSequenceDataset(
            frame, codes, max_history_items=3, target_after_timestamp=2.5)
        self.assertEqual(len(dataset), 2)
        targets = [dataset[index][1].tolist() for index in range(len(dataset))]
        self.assertEqual(targets, [codes[2].tolist(), codes[3].tolist()])


if __name__ == "__main__":
    unittest.main()
