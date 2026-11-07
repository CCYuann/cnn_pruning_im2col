import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from compensation_controls_worker import core, expand_shared, install, quota_mask
from task_loss_mask_search import recalibrate_bn


class CompensationControlsTest(unittest.TestCase):
    def test_independent_filter_quotas(self):
        weights=torch.tensor([[3.,2.,1.],[1.,3.,2.],[2.,1.,3.],[3.,1.,2.]]).reshape(4,3,1,1)
        q=torch.tensor([0,1,2,3])
        mask=quota_mask(weights,q)
        self.assertTrue(torch.equal(mask.flatten(1).sum(1),q))
        self.assertTrue(bool(mask.flatten(1)[1,1]))
        self.assertFalse(bool(mask.flatten(1)[2,1]))

    def test_quota_invalid(self):
        with self.assertRaises(ValueError):
            quota_mask(torch.ones(2,3),torch.tensor([1,4]))

    def test_shared_expansion_and_optimizer(self):
        model=torch.nn.Sequential(torch.nn.Conv2d(3,16,1,bias=False))
        spec=SimpleNamespace(name="0",module=model[0])
        masks={"0":torch.tensor([[True,False,False],[False,True,True]])}
        full=expand_shared(masks,[spec])
        self.assertTrue(torch.equal(full["0"].flatten(1).sum(1),torch.tensor([1]*8+[2]*8)))
        stats=install(model,[spec],full)
        optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9,weight_decay=.0005)
        model(torch.randn(2,3,4,4)).square().sum().backward()
        core.apply_masks_(model,stats,8,include_grad=True)
        optimizer.step()
        core.apply_masks_(model,stats,8)
        self.assertEqual(int(torch.count_nonzero(model[0].weight.detach()[~full["0"]])),0)

    def test_bn_calibration_preserves_parameters_and_training_forward(self):
        torch.manual_seed(4)
        a=torch.nn.Sequential(torch.nn.Conv2d(3,8,1,bias=False),torch.nn.BatchNorm2d(8))
        b=copy.deepcopy(a)
        params={n:p.clone() for n,p in b.named_parameters()}
        recalibrate_bn(b,(torch.randn(64,3,4,4)+3,None))
        for name,p in b.named_parameters():
            self.assertTrue(torch.equal(p,params[name]))
        x=torch.randn(16,3,4,4)
        a.train(); b.train()
        ya,yb=a(x),b(x)
        self.assertTrue(torch.equal(ya,yb))
        ya.square().sum().backward(); yb.square().sum().backward()
        for pa,pb in zip(a.parameters(),b.parameters()):
            self.assertTrue(torch.equal(pa.grad,pb.grad))


if __name__=="__main__":
    unittest.main()
