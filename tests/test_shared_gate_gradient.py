import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from shared_gate_gradient import gate_derivative, matched_random, propose


class SharedGateGradientTests(unittest.TestCase):
    def test_implicit_derivative_matches_explicit_shared_gates(self):
        torch.manual_seed(32)
        shadow = torch.randn(10, 3, 1, 1, dtype=torch.double)
        masks = torch.tensor([[True, False, True], [False, True, False]])
        full = masks.repeat_interleave(8, 0)[:10].reshape_as(shadow)
        adapted = torch.randn_like(shadow) * full
        x = torch.randn(4, 3, 3, 3, dtype=torch.double)
        weight = adapted.clone().requires_grad_()
        F.conv2d(x, weight).square().mean().backward()
        implicit = gate_derivative(weight.detach(), weight.grad, shadow, masks)
        gate = masks.double().clone().requires_grad_()
        expanded = gate.repeat_interleave(8, 0)[:10].reshape_as(shadow)
        base = torch.where(full, adapted, shadow)
        F.conv2d(x, base * expanded).square().mean().backward()
        torch.testing.assert_close(implicit, gate.grad)

    def test_signed_pair_and_empty_full(self):
        masks = torch.tensor([[True, True, False, False], [True, True, True, True], [False, False, False, False]])
        scores = torch.tensor([[2., -1., -3., 4.], [1., 2., 3., 4.], [1., 2., 3., 4.]])
        result, log = propose(scores, masks)
        self.assertEqual(log[0]["remove"], 0)
        self.assertEqual(log[0]["add"], 2)
        self.assertEqual(log[0]["predicted_ce_delta"], -5.)
        self.assertTrue(torch.equal(result.sum(1), masks.sum(1)))
        self.assertEqual(len(log), 1)

    def test_no_improving_pair(self):
        masks = torch.tensor([[True, False]])
        result, log = propose(torch.tensor([[-2., 1.]]), masks)
        self.assertTrue(torch.equal(result, masks))
        self.assertEqual(log, [])

    def test_random_matches_per_cluster_exchange_budget(self):
        masks = torch.tensor([[True, True, False, False], [True, False, True, False]])
        gated = torch.tensor([[False, True, True, False], [True, False, True, False]])
        random = matched_random(masks, gated, 7)
        self.assertTrue(torch.equal((masks & ~random).sum(1), (masks & ~gated).sum(1)))
        self.assertTrue(torch.equal(random.sum(1), masks.sum(1)))
        self.assertTrue(torch.equal(random, matched_random(masks, gated, 7)))


if __name__ == "__main__":
    unittest.main()
