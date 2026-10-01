import unittest
import torch
from ml.train_search_v22 import smooth_pair_loss,protection_loss

class SmoothSupervisionTests(unittest.TestCase):
    def test_teacher_prefers_increasing_incorrect_small_margin(self):
        gap=torch.tensor([-1.],requires_grad=True)
        smooth_pair_loss(gap,torch.tensor([100.])).mean().backward()
        self.assertLess(gap.grad.item(),0)
        self.assertGreater(abs(gap.grad.item()),0)

    def test_protection_penalizes_regression_without_freezing_large_margins(self):
        gap=torch.tensor([-2.,20.,-2.],requires_grad=True)
        loss=protection_loss(gap,torch.tensor([50.,50.,-10.]))
        self.assertAlmostEqual(loss[0].item(),.12,places=6)
        self.assertEqual(loss[1:].tolist(),[0.,0.])
        loss.sum().backward();self.assertLess(gap.grad[0].item(),0)
        self.assertEqual(gap.grad[1:].tolist(),[0.,0.])

    def test_gate_blocks_gradient_only_for_disabled_endpoint(self):
        a=torch.tensor([.1],requires_grad=True);b=torch.tensor([-.1],requires_grad=True)
        gap=.25*a*400-0*b*400
        smooth_pair_loss(gap,torch.tensor([100.])).mean().backward()
        self.assertNotEqual(a.grad.item(),0);self.assertEqual(b.grad.item(),0)
