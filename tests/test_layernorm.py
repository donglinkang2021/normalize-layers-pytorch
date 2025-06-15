import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from normalize import LayerNorm

class TestLayerNorm(unittest.TestCase):
    def test_layernorm_nlp_example(self):
        B, T, D = 3, 10, 3
        x = torch.randn(B, T, D)

        ln_my = LayerNorm(D)
        ln_torch = nn.LayerNorm(D)

        # ln_my.weight.data = ln_torch.weight.data.clone()
        # ln_my.bias.data = ln_torch.bias.data.clone()

        x_my = ln_my(x)
        x_torch = ln_torch(x)

        self.assertEqual(x_my.shape, x_torch.shape)
        self.assertTrue(torch.allclose(x_my, x_torch), f"Outputs not close. MSE: {F.mse_loss(x_my, x_torch)}")

    def test_layernorm_image_example(self):
        N, C, H, W = 3, 4, 2, 2
        x = torch.randn(N, C, H, W)

        ln_my = LayerNorm([C, H, W])
        ln_torch = nn.LayerNorm([C, H, W])

        # ln_my.weight.data = ln_torch.weight.data.clone()
        # ln_my.bias.data = ln_torch.bias.data.clone()

        x_my = ln_my(x)
        x_torch = ln_torch(x)

        self.assertEqual(x_my.shape, x_torch.shape)
        self.assertTrue(torch.allclose(x_my, x_torch), f"Outputs not close. MSE: {F.mse_loss(x_my, x_torch)}")

if __name__ == '__main__':
    unittest.main()
