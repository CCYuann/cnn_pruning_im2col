import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from activation_patch_scores import activation_scores, explicit_sample_moments, patch_moments, rms_from_moments
from fixed_quota_grouping import shared_masks


class ActivationPatchScoresTest(unittest.TestCase):
    def test_padding_stride_dilation_reference(self):
        x=torch.arange(2*3*7*8,dtype=torch.float32).reshape(2,3,7,8)/101
        for kernel,stride,padding,dilation in [(1,1,0,1),(1,2,0,1),(3,1,1,1),(3,2,1,1),(3,1,2,2)]:
            conv=torch.nn.Conv2d(3,8,kernel,stride=stride,padding=padding,dilation=dilation,bias=False)
            actual,count=patch_moments(x,conv)
            expected,other_count=explicit_sample_moments(x,conv)
            self.assertEqual(count,other_count)
            torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)

    def test_batch_accumulation(self):
        x=torch.randn(5,3,7,8,generator=torch.Generator().manual_seed(13))
        conv=torch.nn.Conv2d(3,8,3,padding=1)
        whole,count=patch_moments(x,conv)
        a,ca=patch_moments(x[:2],conv)
        b,cb=patch_moments(x[2:],conv)
        self.assertEqual(count,ca+cb)
        torch.testing.assert_close(rms_from_moments(whole,count),rms_from_moments(a+b,ca+cb),rtol=1e-12,atol=1e-12)

    def test_zero_and_padding_denominator(self):
        conv=torch.nn.Conv2d(1,8,3,padding=1,bias=False)
        total,count=patch_moments(torch.ones(1,1,2,2),conv)
        self.assertEqual(count,4)
        torch.testing.assert_close(total,torch.tensor([1,2,1,2,4,2,1,2,1],dtype=torch.float64))
        total,count=patch_moments(torch.zeros(1,1,2,2),conv)
        self.assertEqual(float(rms_from_moments(total,count).sum()),0.)

    def test_score_changes_ranking_not_quota(self):
        weights=torch.tensor([[2.,1.],[3.,1.]])
        groups=torch.tensor([[0,1]])
        _,l1=shared_masks(weights,groups,1)
        scores=activation_scores(weights,torch.tensor([.1,10.]))
        _,active=shared_masks(scores,groups,1)
        self.assertTrue(bool(l1[:,0].all()))
        self.assertTrue(bool(active[:,1].all()))
        self.assertTrue(torch.equal(l1.sum(1),active.sum(1)))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            patch_moments(torch.zeros(1,4,3,3),torch.nn.Conv2d(4,8,1,groups=4))
        for rms in [torch.tensor([-1.,0.]),torch.tensor([float("nan"),0.]),torch.ones(3)]:
            with self.assertRaises(ValueError):
                activation_scores(torch.ones(8,2),rms)
        with self.assertRaises(ValueError):
            rms_from_moments(torch.ones(2),0)


if __name__ == "__main__":
    unittest.main()
