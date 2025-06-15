import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from normalize import BatchNorm2d

class TestBatchNorm2d(unittest.TestCase):
    def test_batchnorm2d_image_example(self):
        N, C, H, W = 3, 4, 2, 2
        x = torch.randn(N, C, H, W)

        bn_my = BatchNorm2d(C)
        bn_torch = nn.BatchNorm2d(C)

        x_my = bn_my(x)
        x_torch = bn_torch(x)

        self.assertEqual(x_my.shape, x_torch.shape)
        self.assertTrue(torch.allclose(x_my, x_torch), f"Outputs not close. MSE: {F.mse_loss(x_my, x_torch)}")

if __name__ == '__main__':
    unittest.main()