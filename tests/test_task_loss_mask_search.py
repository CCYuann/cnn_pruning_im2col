import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from task_loss_mask_search import accept, assert_quotas, assign, bn_state, measure, recalibrate_bn, restore_buffers


class TaskLossMaskTests(unittest.TestCase):
    def test_acceptance_and_nonfinite(self):
        self.assertTrue(accept(.8, .9, 1., 1., "dual"))
        self.assertFalse(accept(.8, 1.1, 1., 1., "dual"))
        self.assertTrue(accept(.8, 1.1, 1., 1., "search_only"))
        self.assertFalse(accept(1., .9, 1., 1., "dual"))
        self.assertFalse(accept(float("nan"), .9, 1., 1., "dual"))

    def test_quotas_are_per_cluster_not_only_global(self):
        reference = {"conv": torch.tensor([[True, False], [False, True]])}
        assert_quotas(reference, {"conv": ~reference["conv"]})
        with self.assertRaises(ValueError):
            assert_quotas(reference, {"conv": torch.tensor([[True, True], [False, False]])})

    def test_restore_dense_weights_on_rejected_trial(self):
        conv = torch.nn.Conv2d(2, 10, 1, bias=False)
        dense = conv.weight.detach().clone()
        first = torch.tensor([[True, False], [False, True]])
        second = ~first
        assign(conv, dense, first)
        assign(conv, dense, second)
        self.assertTrue(torch.equal(conv.weight[0, 1], dense[0, 1]))
        assign(conv, dense, first)
        self.assertTrue(torch.equal(conv.weight[8, 1], dense[8, 1]))
        self.assertEqual(float(conv.weight[0, 1]), 0.)

    def test_bn_trial_rollback_and_dropout_eval(self):
        model = torch.nn.Sequential(torch.nn.BatchNorm1d(4), torch.nn.Dropout(), torch.nn.Linear(4, 2)).eval()
        before = bn_state(model)
        data = (torch.randn(64, 4) + 5, torch.zeros(64, dtype=torch.long))
        recalibrate_bn(model, data)
        self.assertFalse(model[1].training)
        self.assertFalse(torch.equal(before["0.running_mean"], model[0].running_mean))
        restore_buffers(model, before)
        for name, value in model.named_buffers():
            self.assertTrue(torch.equal(before[name], value))
        self.assertEqual(measure(model, data), measure(model, data))


if __name__ == "__main__":
    unittest.main()
