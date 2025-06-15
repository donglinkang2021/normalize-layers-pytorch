import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add the parent directory to the Python path to import normalize
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from normalize import BatchNorm1d

class TestBatchNorm1d(unittest.TestCase):
    def test_batchnorm1d_nlp_example(self):
        B, T, D = 3, 10, 3
        x = torch.randn(B, T, D)

        bn_my = BatchNorm1d(D)
        bn_torch = nn.BatchNorm1d(D)

        x_my = bn_my(x)
        # PyTorch's BatchNorm1d expects (N, C) or (N, C, L)
        # For (B, T, D) input, if D is features, it should be (B*T, D)
        x_torch = bn_torch(x.reshape(B * T, D)).reshape(B, T, D)

        self.assertEqual(x_my.shape, x_torch.shape)
        self.assertTrue(torch.allclose(x_my, x_torch), f"Outputs not close. MSE: {F.mse_loss(x_my, x_torch)}")

if __name__ == '__main__':
    unittest.main()
