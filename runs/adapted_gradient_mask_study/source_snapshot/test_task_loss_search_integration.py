import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import task_loss_mask_search as search


class ToyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 8, 1, bias=False)
        with torch.no_grad():
            self.conv.weight[:, 0] = 1.
            self.conv.weight[:, 1] = -2.

    def forward(self, images):
        value = self.conv(images).sum((1, 2, 3))
        return torch.stack((value, -value), 1)


class TaskLossIntegrationTests(unittest.TestCase):
    def run_search(self, check_label):
        model = ToyNetwork().eval()
        masks = {"l1": {"conv": torch.tensor([[True, False]])},
                 "mean_repair": {"conv": torch.tensor([[False, True]])},
                 "worst_repair": {"conv": torch.tensor([[False, True]])}}
        images = torch.ones(64, 2, 1, 1)
        data = dict(search=(images, torch.ones(64, dtype=torch.long)),
                    check=(images, torch.full((64,), check_label, dtype=torch.long)),
                    audit=(images, torch.ones(64, dtype=torch.long)))
        spec = SimpleNamespace(name="conv", module=model.conv)
        args = SimpleNamespace(model="toy", seed=0, policy="dual", bn="frozen")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with patch.object(search, "data_splits", return_value=data), patch.object(search, "emit"):
                search.mask_search(model, [spec], args, directory, masks, torch.device("cpu"))
                first = torch.load(directory / "masks.pt", weights_only=False)["loss_gate"]["conv"].clone()
                record = json.loads((directory / "search.json").read_text())
                # A fresh Dense checkpoint is reconstructed on actual process resume.
                fresh = ToyNetwork().eval()
                search.mask_search(fresh, [SimpleNamespace(name="conv", module=fresh.conv)], args, directory, masks, torch.device("cpu"))
                again = torch.load(directory / "masks.pt", weights_only=False)["loss_gate"]["conv"]
                self.assertTrue(torch.equal(first, again))
        return model, first, record

    def test_whole_network_accepts_and_resumes(self):
        model, mask, report = self.run_search(1)
        self.assertTrue(torch.equal(mask, torch.tensor([[False, True]])))
        self.assertEqual(report["accepted_layers"], 1)
        self.assertEqual(float(model.conv.weight[0, 1]), -2.)

    def test_check_veto_restores_mask_and_real_weights(self):
        model, mask, report = self.run_search(0)
        self.assertTrue(torch.equal(mask, torch.tensor([[True, False]])))
        self.assertEqual(report["accepted_layers"], 0)
        self.assertEqual(float(model.conv.weight[0, 0]), 1.)
        self.assertEqual(float(model.conv.weight[0, 1]), 0.)


if __name__ == "__main__":
    unittest.main()
