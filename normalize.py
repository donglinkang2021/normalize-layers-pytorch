import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Union, List

_size = Union[int, List[int]]

def normalize(x:torch.Tensor, mu:torch.Tensor, var:torch.Tensor):
    return (x - mu) * var.rsqrt()

# for both nlp and image tasks
# x shape: (B, T, D) for nlp tasks
# x shape: (N, C, H, W) for image tasks

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape:_size, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.weight = nn.Parameter(torch.ones(*normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(*normalized_shape), requires_grad=True)
        self.eps = eps

    def forward(self, x:torch.Tensor):
        # according to the self.normalized_shape to decide the dimensions to normalize
        dims = tuple(range(-len(self.normalized_shape), 0))
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False) + self.eps
        x_norm = normalize(x, mean, var)
        return x_norm * self.weight + self.bias

class RMSNorm(nn.Module):
    def __init__(self, normalized_shape:_size, eps=1e-7):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.weight = nn.Parameter(torch.ones(*normalized_shape), requires_grad=True)
        self.eps = eps

    def forward(self, x:torch.Tensor):
        # according to the self.normalized_shape to decide the dimensions to normalize
        dims = tuple(range(-len(self.normalized_shape), 0))
        var = x.pow(2).mean(dim=dims, keepdim=True) + self.eps
        x_norm = x * var.rsqrt()
        return x_norm * self.weight

class DyT(nn.Module):
    """https://jiachenzhu.github.io/DyT/"""
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value, requires_grad=True)
        self.weight = nn.Parameter(torch.ones(num_features), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(num_features), requires_grad=True)
    
    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias


# only for nlp tasks
# x shape: (B, T, D)

class BatchNorm1d(nn.Module):
    def __init__(self, num_features:int, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(num_features), requires_grad=True)
        self.eps = eps

    def forward(self, x:torch.Tensor):
        B, T, D = x.shape
        x = rearrange(x, 'b t d -> (b t) d')
        mean = x.mean(dim=0, keepdim=True)
        var = x.var(dim=0, keepdim=True, unbiased=False) + self.eps
        x_norm = normalize(x, mean, var)
        x_norm = rearrange(x_norm, ' (b t) d -> b t d', b=B)
        return x_norm * self.weight + self.bias

# only for image tasks
# x shape: (N, C, H, W)

class BatchNorm2d(nn.Module):
    def __init__(self, num_features:int, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1,num_features,1,1), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(1,num_features,1,1), requires_grad=True)
        self.eps = eps

    def forward(self, x:torch.Tensor):
        N, C, H, W = x.shape
        x = rearrange(x, 'n c h w -> (n h w) c')
        mean = x.mean(dim=0, keepdim=True)
        var = x.var(dim=0, keepdim=True, unbiased=False) + self.eps
        x_norm = normalize(x, mean, var)
        x_norm = rearrange(x_norm, '(n h w) c -> n c h w', n=N, h=H, w=W)
        return x_norm * self.weight + self.bias

class GroupNorm(nn.Module):
    def __init__(self, num_groups:int, num_channels:int, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1), requires_grad=True)
        self.eps = eps

    def forward(self, x:torch.Tensor):
        N, C, H, W = x.shape
        assert C % self.num_groups == 0, "Number of channels must be divisible by num_groups"
        x = x.view(N, self.num_groups, -1)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False) + self.eps
        x_norm = normalize(x, mean, var)
        x_norm = x_norm.view(N, C, H, W)
        return x_norm * self.weight + self.bias

def _test_batchnorm1d():
    # nlp example
    B, T, D = 3, 10, 3
    x = torch.randn(B, T, D)

    bn_my = BatchNorm1d(D)
    bn_torch = nn.BatchNorm1d(D)

    x_my = bn_my(x)
    x_torch = bn_torch(x.reshape(B * T, D)).reshape(B, T, D)
    print(x_my.shape, x_torch.shape)
    print("x_my[0]:", x_my[0])
    print("x_torch[0]:", x_torch[0])
    print(torch.allclose(x_my, x_torch))
    print("MSE:", F.mse_loss(x_my, x_torch))

    print("BatchNorm1d test passed.")

def _test_batchnorm2d():
    # image example
    N, C, H, W = 3, 4, 2, 2
    x = torch.randn(N, C, H, W)

    bn_my = BatchNorm2d(C)
    bn_torch = nn.BatchNorm2d(C)

    x_my = bn_my(x)
    x_torch = bn_torch(x)
    print(x_my.shape, x_torch.shape)
    print("x_my[0]:", x_my[0])
    print("x_torch[0]:", x_torch[0])
    print(torch.allclose(x_my, x_torch))
    print("MSE:", F.mse_loss(x_my, x_torch))

    print("BatchNorm2d test passed.")

def _test_layernorm():
    # nlp example
    B, T, D = 3, 10, 3
    x = torch.randn(B, T, D)

    ln_my = LayerNorm(D)
    ln_torch = nn.LayerNorm(D)

    x_my = ln_my(x)
    x_torch = ln_torch(x)
    print(x_my.shape, x_torch.shape)
    print("x_my[0]:", x_my[0])
    print("x_torch[0]:", x_torch[0])
    print(torch.allclose(x_my, x_torch))
    print("MSE:", F.mse_loss(x_my, x_torch))

    # image example
    N, C, H, W = 3, 4, 2, 2
    x = torch.randn(N, C, H, W)
    ln_my = LayerNorm([C, H, W])
    ln_torch = nn.LayerNorm([C, H, W])
    x_my = ln_my(x)
    x_torch = ln_torch(x)
    print(x_my.shape, x_torch.shape)
    print("x_my[0]:", x_my[0])
    print("x_torch[0]:", x_torch[0])
    print(torch.allclose(x_my, x_torch))
    print("MSE:", F.mse_loss(x_my, x_torch))

    print("LayerNorm test passed.")

def _test_rmsnorm():
    # nlp example
    B, T, D = 3, 10, 3
    x = torch.randn(B, T, D)

    rms_my = RMSNorm(D)
    rms_torch = nn.RMSNorm(D)  # RMSNorm is similar to LayerNorm

    x_my = rms_my(x)
    x_torch = rms_torch(x)
    print(x_my.shape, x_torch.shape)
    print("x_my[0]:", x_my[0])
    print("x_torch[0]:", x_torch[0])
    print(torch.allclose(x_my, x_torch))
    print("MSE:", F.mse_loss(x_my, x_torch))

    # image example
    N, C, H, W = 3, 4, 2, 2
    x = torch.randn(N, C, H, W)
    rms_my = RMSNorm([C, H, W])
    rms_torch = nn.RMSNorm([C, H, W])
    x_my = rms_my(x)
    x_torch = rms_torch(x)
    print(x_my.shape, x_torch.shape)
    print("x_my[0]:", x_my[0])
    print("x_torch[0]:", x_torch[0])
    print(torch.allclose(x_my, x_torch))
    print("MSE:", F.mse_loss(x_my, x_torch))

    print("RMSNorm test passed.")

def _test_groupnorm():
    # image example
    N, C, H, W = 3, 4, 2, 2
    x = torch.randn(N, C, H, W)

    gn_my = GroupNorm(num_groups=2, num_channels=C, eps=1e-5)
    gn_torch = nn.GroupNorm(num_groups=2, num_channels=C)

    x_my = gn_my(x)
    x_torch = gn_torch(x)

    print(x_my.shape, x_torch.shape)
    print("x_my[0]:", x_my[0])
    print("x_torch[0]:", x_torch[0])
    print(torch.allclose(x_my, x_torch))
    print("MSE:", F.mse_loss(x_my, x_torch))
    print("GroupNorm test passed.")


if __name__ == "__main__":
    
    # _test_batchnorm1d()
    # _test_batchnorm2d()
    # _test_layernorm()
    _test_rmsnorm()
    # _test_groupnorm()

