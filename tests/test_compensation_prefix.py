import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from fixed_support_ridge import positions_for_layer
from shared_mask_repair import sample_patches
from weight_compensation_worker import student_patches


class CompensationPrefixTests(unittest.TestCase):
    def test_actual_prefix_changes_and_bn_is_unchanged(self):
        torch.manual_seed(17)
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1, bias=False), torch.nn.BatchNorm2d(4),
                                    torch.nn.ReLU(), torch.nn.Conv2d(4, 8, 3, padding=1, bias=False)).eval()
        images = torch.randn(40, 3, 6, 6)
        positions = positions_for_layer(40, 36, 320, 11)
        bn_before = {n:v.clone() for n,v in model[1].state_dict().items()}
        actual = student_patches(model, model[3], images, positions, torch.device("cpu"))
        expected = sample_patches(model[:3](images), model[3], positions)
        torch.testing.assert_close(actual, expected)
        with torch.no_grad():
            model[0].weight.mul_(2)
        changed = student_patches(model, model[3], images, positions, torch.device("cpu"))
        self.assertFalse(torch.equal(actual, changed))
        torch.testing.assert_close(changed, actual*2)
        for name, value in model[1].state_dict().items():
            self.assertTrue(torch.equal(value, bn_before[name]))


if __name__ == "__main__":
    unittest.main()
