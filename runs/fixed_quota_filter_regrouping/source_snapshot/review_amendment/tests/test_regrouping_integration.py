import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from fixed_quota_grouping import exchange_search, grouping_loss, shared_masks
from fixed_quota_grouping_worker import core, install_masks
from adapted_gradient_mask import restore_rng, rng_state
from run_fixed_quota_regrouping import mean_or_none


class RegroupingIntegrationTest(unittest.TestCase):
    def test_multiple_disjoint_exchanges(self):
        generator=torch.Generator().manual_seed(19)
        weights=torch.rand(32,27,generator=generator)
        initial=torch.arange(32).reshape(-1,8)
        groups,report=exchange_search(weights,initial,9,3)
        self.assertTrue(any(r["accepted"]>1 for r in report["rounds"]))
        self.assertAlmostEqual(report["after"],float(grouping_loss(weights,groups,9)))
        self.assertLess(report["after"],report["before"])

    def test_noncontiguous_masks_survive_optimizer_and_resume(self):
        torch.manual_seed(11)
        model=torch.nn.Sequential(torch.nn.Conv2d(3,16,1,bias=False),torch.nn.BatchNorm2d(16))
        groups=torch.tensor([[0,2,4,6,8,10,12,14],[1,3,5,7,9,11,13,15]])
        masks,full=shared_masks(model[0].weight.detach().flatten(1),groups,1)
        pack={"0":dict(groups=groups,masks=masks,q=1)}
        stats=install_masks(model,[SimpleNamespace(name="0",module=model[0])],pack)
        self.assertTrue(torch.equal(stats[0].full_mask.flatten(1),full))
        optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9,weight_decay=.0005)
        loader=SimpleNamespace(generator=torch.Generator().manual_seed(8))
        def step(net,opt,mask_stats):
            opt.zero_grad(set_to_none=True)
            x=torch.randn(4,3,2,2)+torch.rand((),generator=loader.generator)
            net(x).square().sum().backward()
            core.apply_masks_(net,mask_stats,8,include_grad=True)
            opt.step()
            core.apply_masks_(net,mask_stats,8)
        step(model,optimizer,stats)
        saved=copy.deepcopy(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng_state(loader)))
        step(model,optimizer,stats)
        expected=copy.deepcopy(model.state_dict())
        resumed=copy.deepcopy(model)
        resumed.load_state_dict(saved["model"])
        newopt=torch.optim.SGD(resumed.parameters(),lr=.01,momentum=.9,weight_decay=.0005)
        newopt.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"],loader)
        step(resumed,newopt,stats)
        for name,value in resumed.state_dict().items():
            self.assertTrue(torch.equal(value,expected[name]),name)
        self.assertEqual(int(torch.count_nonzero(resumed[0].weight.detach().flatten(1)[~full])),0)

    def test_zero_gap_aggregate(self):
        self.assertIsNone(mean_or_none([None,None]))
        self.assertEqual(mean_or_none([None,.1,.3]),.2)


if __name__ == "__main__":
    unittest.main()
