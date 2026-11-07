import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from shared_mask_repair import cluster_inputs, metrics, project, repair, sample_patches


class SharedMaskTests(unittest.TestCase):
    def test_patch_order_stride_padding_dilation(self):
        for kernel, stride, padding, dilation in [(1,1,0,1),(3,2,1,1),(3,1,2,2)]:
            conv = torch.nn.Conv2d(3, 8, kernel, stride=stride, padding=padding, dilation=dilation)
            value = torch.randn(2,3,9,9)
            all_patches = F.unfold(value, conv.kernel_size, dilation=conv.dilation, padding=conv.padding, stride=conv.stride)
            positions = torch.tensor([[0,3],[1,4]])
            expected = torch.stack([all_patches[i,:,j] for i in range(2) for j in positions[i]],dim=1)
            torch.testing.assert_close(sample_patches(value,conv,positions),expected)

    def test_grouped_and_depthwise_input_mapping(self):
        for groups in [1,2,4]:
            conv=torch.nn.Conv2d(4,8,3,padding=1,groups=groups,bias=False)
            value=torch.randn(2,4,5,5)
            positions=torch.tensor([[0,3],[1,4]])
            patches=sample_patches(value,conv,positions)
            inputs=cluster_inputs(patches,conv,0,8)
            actual=project(conv.weight.flatten(1),inputs)
            output=conv(value).flatten(2)
            expected=torch.stack([output[i,:,j] for i in range(2) for j in positions[i]],1)
            torch.testing.assert_close(actual,expected)

    def test_toy_conflict_improves_worst_and_keeps_budget(self):
        weight=torch.tensor([[10.,9,0,0],[0,0,1,1]])
        inputs=torch.eye(4).unsqueeze(0)
        mask=torch.tensor([True,True,False,False])
        result,swaps=repair(weight,inputs,mask,1e-8)
        self.assertEqual(int(result.sum()),2)
        self.assertGreater(len(swaps),0)
        before,after=metrics(weight,inputs,mask,1e-8),metrics(weight,inputs,result,1e-8)
        self.assertLess(after['max_error'],before['max_error'])
        self.assertLessEqual(after['mean_error'],before['mean_error'])

    def test_random_and_grouped_swap_predictions(self):
        for grouped in [False,True]:
            torch.manual_seed(32)
            weight=torch.randn(8,27)
            inputs=torch.randn(8 if grouped else 1,27,128)
            mask=torch.arange(27)<4
            result,_=repair(weight,inputs,mask,1e-6)
            self.assertEqual(int(result.sum()),4)
            self.assertLessEqual(metrics(weight,inputs,result,1e-6)['mean_error'],metrics(weight,inputs,mask,1e-6)['mean_error']+1e-6)

    def test_empty_full_and_zero_energy(self):
        weight=torch.zeros(2,4)
        inputs=torch.eye(4).unsqueeze(0)
        for mask in [torch.zeros(4,dtype=torch.bool),torch.ones(4,dtype=torch.bool)]:
            result,swaps=repair(weight,inputs,mask,1e-6)
            self.assertTrue(torch.equal(mask,result))
            self.assertEqual(swaps,[])

    def test_two_pass_preserves_post_mean_error_bound(self):
        torch.manual_seed(93)
        weight=torch.randn(8,27)
        inputs=torch.randn(1,27,256)
        mask=torch.arange(27)<4
        intermediate,_=repair(weight,inputs,mask,1e-6,"mean",8,4)
        result,_=repair(weight,inputs,intermediate,1e-6,"worst",8,4)
        before=metrics(weight,inputs,intermediate,1e-6)
        after=metrics(weight,inputs,result,1e-6)
        self.assertLessEqual(after['mean_error'],before['mean_error']+1e-6)
        self.assertLessEqual(after['max_error'],before['max_error']+1e-6)
        self.assertEqual(int(result.sum()),4)


if __name__ == '__main__':
    unittest.main()
