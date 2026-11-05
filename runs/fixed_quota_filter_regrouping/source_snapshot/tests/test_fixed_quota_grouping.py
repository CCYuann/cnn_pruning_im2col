import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from fixed_quota_grouping import closure, discarded, exchange_search, grouping_loss, shared_masks, validate_groups


class FixedQuotaGroupingTest(unittest.TestCase):
    def setUp(self):
        self.w = torch.tensor([[9.,8,0,0], [0.,0,9,8], [8.,9,0,0], [0.,0,8,9]])

    def test_toy_cost_and_lower_bound(self):
        contiguous = torch.tensor([[0,1],[2,3]])
        similar = torch.tensor([[0,2],[1,3]])
        self.assertEqual(float(grouping_loss(self.w, contiguous, 2)), 32.)
        self.assertEqual(float(grouping_loss(self.w, similar, 2)), 0.)
        lower = float(discarded(self.w, 2).sum())
        self.assertEqual(closure(32., 0., lower), 1.)

    def test_noncontiguous_scatter(self):
        groups = torch.tensor([[0,2],[1,3]])
        masks, full = shared_masks(self.w, groups, 2)
        self.assertTrue(torch.equal(full[0], masks[0]))
        self.assertTrue(torch.equal(full[2], masks[0]))
        self.assertTrue(torch.equal(full[1], masks[1]))
        self.assertTrue(torch.equal(full[3], masks[1]))
        self.assertEqual(int((self.w * ~full).sum()), 0)

    def test_exchange_and_determinism(self):
        initial = torch.tensor([[0,1],[2,3]])
        a, report = exchange_search(self.w, initial, 2, 13)
        b, other = exchange_search(self.w, initial, 2, 13)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(report, other)
        self.assertEqual(report["after"], 0.)
        self.assertTrue(torch.equal(initial, torch.tensor([[0,1],[2,3]])))

    def test_empty_full_and_zero_gap(self):
        groups = torch.tensor([[0,1],[2,3]])
        for q in (0, 4):
            masks, full = shared_masks(self.w, groups, q)
            self.assertTrue(bool((full.sum(1) == q).all()))
            _, report = exchange_search(self.w, groups, q, 13)
            self.assertEqual(report["accepted"], 0)
        self.assertIsNone(closure(0., 0., 0.))
        self.assertLess(closure(2., 3., 0.), 0.)

    def test_invalid_partition(self):
        with self.assertRaises(ValueError):
            validate_groups(torch.tensor([[0,1],[1,3]]), 4, 2)

    def test_random_mask_matches_objective(self):
        generator = torch.Generator().manual_seed(4)
        weights = torch.randn(16, 27, generator=generator)
        groups = torch.randperm(16, generator=generator).reshape(-1, 8)
        for q in (1, 13, 26):
            _, full = shared_masks(weights, groups, q)
            torch.testing.assert_close((weights.double().abs()*~full).sum(), grouping_loss(weights, groups, q))


if __name__ == "__main__":
    unittest.main()
