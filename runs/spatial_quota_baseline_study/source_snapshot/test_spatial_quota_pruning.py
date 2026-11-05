import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from spatial_quota_pruning import cluster_masks, spatial_counts


class SpatialQuotaTests(unittest.TestCase):
    def test_largest_remainder_and_raster_tie_break(self):
        reference = torch.tensor([[[[1, 1, 1]]], [[[0, 0, 0]]]], dtype=torch.bool)
        counts, ideal = spatial_counts(reference, 1)
        self.assertTrue(torch.equal(counts, torch.tensor([[1, 0, 0]])))
        self.assertAlmostEqual(float(ideal.sum()), 1.5)

    def test_all_empty_or_full(self):
        for value in [False, True]:
            reference = torch.full((8, 5, 3, 3), value)
            counts, _ = spatial_counts(reference, 45 if value else 0)
            self.assertTrue(torch.equal(counts, torch.full((3, 3), 5 if value else 0)))

    def test_pointwise_identical_even_with_tied_importance(self):
        for criterion in ["l1_sum", "top2_mean_abs"]:
            for channels in [1, 3, 64]:
                for width in [3, 8]:
                    weight = torch.ones(width, channels, 1, 1)
                    reference = torch.rand_like(weight) > .8
                    flat, spatial, _, _ = cluster_masks(weight, reference, criterion)
                    self.assertTrue(torch.equal(flat, spatial))

    def test_fuzz_quotas_and_kernel_order(self):
        torch.manual_seed(199)
        for channels in [1, 3, 9, 64]:
            for kernel in [1, 3, 5]:
                weight = torch.randn(8, channels, kernel, kernel)
                reference = torch.rand_like(weight) > .85
                for criterion in ["l1_sum", "top2_mean_abs"]:
                    flat, spatial, counts, _ = cluster_masks(weight, reference, criterion)
                    self.assertEqual(int(flat.sum()), int(spatial.sum()))
                    self.assertTrue(torch.equal(spatial.sum(0).cpu(), counts))
                    score = weight.abs().sum(0) if criterion == "l1_sum" else weight.abs().topk(2, dim=0).values.mean(0)
                    for r in range(kernel):
                        for s in range(kernel):
                            kept = spatial[:, r, s]
                            if kept.any() and (~kept).any():
                                self.assertGreaterEqual(float(score[:, r, s][kept].min()), float(score[:, r, s][~kept].max()))

    def test_spatial_can_differ_without_changing_budget(self):
        weight = torch.tensor([[[[10., 1.]], [[9., .5]]]]).repeat(8, 1, 1, 1)
        reference = torch.tensor([[[[1, 1]], [[0, 0]]]]).repeat(8, 1, 1, 1)
        flat, spatial, _, _ = cluster_masks(weight, reference, "l1_sum")
        self.assertFalse(torch.equal(flat, spatial))
        self.assertEqual(int(flat.sum()), 2)
        self.assertEqual(int(spatial.sum()), 2)

    def test_invalid_reference_or_budget_rejected(self):
        with self.assertRaises(ValueError):
            spatial_counts(torch.ones(8, 3, 3), 2)
        with self.assertRaises(ValueError):
            spatial_counts(torch.ones(8, 3, 3, 3), 0)
        with self.assertRaises(ValueError):
            spatial_counts(torch.full((8, 3, 3, 3), .5), 13)


if __name__ == "__main__":
    unittest.main()
