import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from scalar_weight_compensation import compensate_scalar


class ScalarWeightCompensationTest(unittest.TestCase):
    def test_exact_per_filter_scaling(self):
        w=torch.tensor([[1.,2.],[2.,3.]])
        x=torch.tensor([[1.,2.,3.],[0.,0.,0.]])
        y=torch.tensor([[2.,4.,6.],[-2.,-4.,-6.]])
        mask=torch.tensor([True,False])
        value,report=compensate_scalar(w,mask,x,y,x,y)
        torch.testing.assert_close(value,torch.tensor([[2.,0.],[-2.,0.]]))
        self.assertEqual(report["status"],"accepted")
        self.assertEqual(report["alpha"],[2.,-1.])

    def test_check_fallback(self):
        w=torch.tensor([[1.,2.]])
        x=torch.tensor([[1.,2.],[0.,0.]])
        mask=torch.tensor([True,False])
        value,report=compensate_scalar(w,mask,x,3*x[:1],x,x[:1])
        self.assertEqual(report["status"],"fallback")
        torch.testing.assert_close(value,torch.tensor([[1.,0.]]))

    def test_empty_full_and_zero_energy(self):
        w=torch.tensor([[1.,2.]])
        x=torch.zeros(2,3)
        for mask,status in [(torch.tensor([False,False]),"empty"),(torch.tensor([True,True]),"full"),(torch.tensor([True,False]),"accepted")]:
            value,report=compensate_scalar(w,mask,x,torch.zeros(1,3),x,torch.zeros(1,3))
            self.assertEqual(report["status"],status)
            self.assertEqual(report["alpha"],[1.])
            self.assertEqual(int(torch.count_nonzero(value[:,~mask])),0)

    def test_invalid_alignment(self):
        with self.assertRaises(ValueError):
            compensate_scalar(torch.ones(2,3),torch.ones(3,dtype=torch.bool),torch.ones(3,4),torch.ones(2,3),torch.ones(3,4),torch.ones(2,4))


if __name__=="__main__":
    unittest.main()
