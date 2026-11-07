import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from adapted_gradient_mask import build_stats, restore_rng, rng_state, update_masks


class AdaptedGradientTests(unittest.TestCase):
    def test_swap_restores_shadow_and_clears_new_momentum(self):
        conv = torch.nn.Conv2d(2, 8, 1, bias=False)
        model = torch.nn.Sequential(conv)
        with torch.no_grad():
            conv.weight[:, 0] = 5.
            conv.weight[:, 1] = 0.
        initial = {"0": torch.tensor([[True, False]])}
        selected = {"0": torch.tensor([[False, True]])}
        shadow = {"0": torch.ones_like(conv.weight)*2}
        specs = [SimpleNamespace(name="0", module=conv, input_hw=(1, 1))]
        stats = build_stats(specs, initial)
        optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=.9)
        optimizer.state[conv.weight]["momentum_buffer"] = torch.ones_like(conv.weight)*3
        update_masks(model, specs, stats, initial, selected, shadow, optimizer)
        self.assertEqual(float(conv.weight[0, 0]), 0.)
        self.assertEqual(float(conv.weight[0, 1]), 2.)
        self.assertEqual(float(optimizer.state[conv.weight]["momentum_buffer"].abs().sum()), 0.)

    def test_unchanged_connections_keep_momentum(self):
        conv = torch.nn.Conv2d(2, 8, 1, bias=False)
        model = torch.nn.Sequential(conv)
        mask = {"0": torch.tensor([[True, False]])}
        specs = [SimpleNamespace(name="0", module=conv, input_hw=(1, 1))]
        stats = build_stats(specs, mask)
        before = conv.weight.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=.9)
        optimizer.state[conv.weight]["momentum_buffer"] = torch.ones_like(conv.weight)*3
        update_masks(model, specs, stats, mask, mask, {"0": torch.ones_like(conv.weight)}, optimizer)
        self.assertTrue(torch.equal(before, conv.weight))
        self.assertEqual(float(optimizer.state[conv.weight]["momentum_buffer"][0, 0]), 3.)
        self.assertEqual(float(optimizer.state[conv.weight]["momentum_buffer"][0, 1]), 0.)

    def test_search_does_not_change_training_rng_stream(self):
        loader = SimpleNamespace(generator=torch.Generator().manual_seed(39))
        state = rng_state(loader)
        def draws():
            return (random.random(), float(np.random.rand()), float(torch.rand(())),
                    float(torch.rand((), generator=loader.generator)))
        expected = draws()
        for _ in range(10):
            draws()
        restore_rng(state, loader)
        self.assertEqual(expected, draws())


if __name__ == "__main__":
    unittest.main()
