import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from normalize import RMSNorm

class TestRMSNorm(unittest.TestCase):
    def test_rmsnorm_nlp_example(self):
        B, T, D = 3, 10, 3
        x = torch.randn(B, T, D)

        rms_my = RMSNorm(D)
        # Assuming nn.RMSNorm exists and works similarly for comparison
        # If not, this part of the test might need adjustment or a different comparison target
        try:
            rms_torch = nn.RMSNorm(D)
            # rms_my.weight.data = rms_torch.weight.data.clone()
            x_my = rms_my(x)
            x_torch = rms_torch(x)
            self.assertEqual(x_my.shape, x_torch.shape)
            self.assertTrue(torch.allclose(x_my, x_torch), f"Outputs not close. MSE: {F.mse_loss(x_my, x_torch)}")
        except AttributeError:
            print("nn.RMSNorm not available in this PyTorch version, skipping direct torch comparison for NLP.")
            # Basic check for shape and execution
            x_my = rms_my(x)
            self.assertEqual(x_my.shape, (B,T,D))


    def test_rmsnorm_image_example(self):
        N, C, H, W = 3, 4, 2, 2
        x = torch.randn(N, C, H, W)

        rms_my = RMSNorm([C, H, W])
        try:
            rms_torch = nn.RMSNorm([C, H, W])
            # rms_my.weight.data = rms_torch.weight.data.clone()
            x_my = rms_my(x)
            x_torch = rms_torch(x)
            self.assertEqual(x_my.shape, x_torch.shape)
            self.assertTrue(torch.allclose(x_my, x_torch), f"Outputs not close. MSE: {F.mse_loss(x_my, x_torch)}")
        except AttributeError:
            print("nn.RMSNorm not available in this PyTorch version, skipping direct torch comparison for Image.")
            # Basic check for shape and execution
            x_my = rms_my(x)
            self.assertEqual(x_my.shape, (N,C,H,W))


if __name__ == '__main__':
    unittest.main()
