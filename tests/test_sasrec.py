# -*- coding: utf-8 -*-
"""Smoke tests for the SASRec sequence model plumbing.

Covers the left-padding helper and the single most load-bearing architectural
invariant: causal (unidirectional) attention. Both run on CPU with a tiny model
in ``eval`` mode, so no GPU, checkpoint, or dataset is required.
"""
import unittest

import torch

from train_sasrec import SASRec, pad_left


class PadLeftTests(unittest.TestCase):
    def test_pads_on_the_left(self):
        self.assertEqual(pad_left([1, 2, 3], 5), [0, 0, 1, 2, 3])

    def test_truncates_to_most_recent(self):
        self.assertEqual(pad_left([1, 2, 3, 4, 5, 6], 3), [4, 5, 6])

    def test_exact_length_is_unchanged(self):
        self.assertEqual(pad_left([7, 8, 9], 3), [7, 8, 9])

    def test_output_length_always_equals_L(self):
        for L in (1, 4, 8):
            self.assertEqual(len(pad_left([1, 2, 3], L)), L)


class SASRecCausalityTests(unittest.TestCase):
    def _tiny_model(self):
        torch.manual_seed(0)
        model = SASRec(n_items=20, d=16, max_len=8, n_blocks=2, dropout=0.0)
        model.eval()  # freeze dropout so the forward pass is deterministic
        return model

    def test_encode_shape(self):
        model = self._tiny_model()
        seq = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            out = model.encode(seq)
        self.assertEqual(tuple(out.shape), (1, 4, 16))

    def test_future_tokens_do_not_leak_into_the_past(self):
        """Changing the last token must not alter earlier positions' context."""
        model = self._tiny_model()
        a = torch.tensor([[1, 2, 3, 4]])
        b = torch.tensor([[1, 2, 3, 9]])  # differ only at the final position
        with torch.no_grad():
            ctx_a = model.encode(a)
            ctx_b = model.encode(b)
        # Positions 0..2 see only tokens <= their index -> must be identical.
        self.assertTrue(torch.allclose(ctx_a[:, :3], ctx_b[:, :3], atol=1e-6))
        # Sanity: the final position, which sees the changed token, differs.
        self.assertFalse(torch.allclose(ctx_a[:, 3], ctx_b[:, 3], atol=1e-6))

    def test_padding_positions_are_zeroed(self):
        model = self._tiny_model()
        seq = torch.tensor([[0, 0, 5, 6]])  # left-padded with the pad id 0
        with torch.no_grad():
            out = model.encode(seq)
        self.assertTrue(torch.allclose(out[0, :2], torch.zeros(2, 16), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
