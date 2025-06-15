import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from normalize import GroupNorm

class TestGroupNorm(unittest.TestCase):
    def test_groupnorm_image_example(self):
        N, C, H, W = 3, 4, 2, 2
        num_groups = 2
        x = torch.randn(N, C, H, W)

        # Ensure C is divisible by num_groups for the test
        self.assertEqual(C % num_groups, 0, "Number of channels must be divisible by num_groups for this test setup.")

        gn_my = GroupNorm(num_groups=num_groups, num_channels=C)
        gn_torch = nn.GroupNorm(num_groups=num_groups, num_channels=C) # Match eps for closer comparison

        # gn_my.weight.data = gn_torch.weight.data.clone().reshape_as(gn_my.weight.data)
        # gn_my.bias.data = gn_torch.bias.data.clone().reshape_as(gn_my.bias.data)

        x_my = gn_my(x)
        x_torch = gn_torch(x)

        self.assertEqual(x_my.shape, x_torch.shape)
        # GroupNorm can have slight differences due to implementation details, adjust atol if needed
        self.assertTrue(torch.allclose(x_my, x_torch), f"Outputs not close. MSE: {F.mse_loss(x_my, x_torch)}")

if __name__ == '__main__':
    unittest.main()
