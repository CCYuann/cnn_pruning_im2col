import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from fixed_support_ridge import compensate, positions_for_layer, sampled_output
from shared_mask_repair import sample_patches


class FixedSupportRidgeTests(unittest.TestCase):
    def test_prior_centered_solution_matches_augmented_lstsq(self):
        torch.manual_seed(71)
        w = torch.randn(3, 5, dtype=torch.double)
        mask = torch.tensor([True, False, True, False, True])
        x = torch.randn(5, 30, dtype=torch.double)
        y = w @ x
        result, report = compensate(w, mask, x, y, x, y)
        xs, w0 = x[mask], w[:, mask]
        lam = report["lambda_base"]
        a = torch.cat((xs.T/(30**.5), torch.eye(3, dtype=torch.double)*lam**.5))
        b = torch.cat((y.T/(30**.5), w0.T*lam**.5))
        expected = torch.linalg.lstsq(a, b).solution.T
        torch.testing.assert_close(result[:, mask], expected)
        self.assertEqual(int(torch.count_nonzero(result[:, ~mask])), 0)

    def test_rank_deficient_input_is_regularized(self):
        w = torch.ones(2, 4)
        x = torch.ones(4, 8)
        result, report = compensate(w, torch.tensor([True, True, False, False]), x, w@x, x, w@x)
        self.assertTrue(torch.isfinite(result).all())
        self.assertLess(report["fit_mse_after"], report["fit_mse_before"])

    def test_empty_and_full_keep_definition(self):
        w, x = torch.randn(2, 4), torch.randn(4, 10)
        for mask in [torch.zeros(4, dtype=torch.bool), torch.ones(4, dtype=torch.bool)]:
            result, _ = compensate(w, mask, x, w@x, x, w@x)
            self.assertTrue(torch.equal(result, w*mask))

    def test_check_regression_falls_back_without_mask_change(self):
        w = torch.tensor([[1., 1.]])
        mask = torch.tensor([True, False])
        x = torch.ones(2, 10)
        result, report = compensate(w, mask, x, torch.ones(1, 10)*2, x, torch.ones(1, 10))
        self.assertEqual(report["status"], "fallback")
        self.assertTrue(torch.equal(result, w*mask))

    def test_output_coordinate_and_k_order(self):
        for kernel, stride, padding in [(1, 1, 0), (3, 2, 1), (3, 1, 1)]:
            conv = torch.nn.Conv2d(3, 8, kernel, stride=stride, padding=padding, bias=False).double()
            image = torch.randn(4, 3, 9, 9, dtype=torch.double)
            output = conv(image)
            positions = positions_for_layer(4, output.shape[-2]*output.shape[-1], 16, 123)
            actual = sampled_output(output, positions)
            expected = conv.weight.flatten(1) @ sample_patches(image, conv, positions)
            torch.testing.assert_close(actual, expected)

    def test_unique_deterministic_sampling(self):
        a = positions_for_layer(4, 20, 32, 99)
        self.assertTrue(torch.equal(a, positions_for_layer(4, 20, 32, 99)))
        self.assertTrue(all(len(row.unique()) == len(row) for row in a))

    def test_nonfinite_and_misaligned_inputs_rejected(self):
        w = torch.ones(2, 4)
        mask = torch.tensor([True, False, True, False])
        x, y = torch.ones(4, 10), torch.ones(2, 10)
        with self.assertRaises(ValueError):
            compensate(w, mask, x, y[:, :9], x, y)
        x[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            compensate(w, mask, x, y, x, y)


if __name__ == "__main__":
    unittest.main()
