"""BitLinear Layer — Ternary (1.58-bit) quantization for PyTorch.

Based on "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits" (2024).
Weights are quantized to {-1, 0, 1} for extreme efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_quant(w):
    """Quantize weights to {-1, 0, 1} using absmean scaling."""
    scale = w.abs().mean().clamp(min=1e-5)
    w_q = torch.round(torch.clamp(w / scale, -1, 1))
    # Straight-Through Estimator (STE)
    return w + (w_q - w).detach()


def activation_quant(x):
    """Quantize activations to 8-bit using absmax scaling."""
    scale = 127.0 / x.abs().max().clamp(min=1e-5)
    x_q = torch.round(torch.clamp(x * scale, -128, 127))
    # Straight-Through Estimator (STE)
    return x + (x_q / scale - x).detach()


class BitLinear(nn.Linear):
    """Linear layer with ternary weights and 8-bit activations."""

    def forward(self, x):
        # Quantize weights
        w_q = weight_quant(self.weight)
        
        # Quantize activations (optional for small models, but good for BitNet adherence)
        x_q = activation_quant(x)
        
        # Perform linear operation (Matrix Multiplication)
        # Note: In a real C++ kernel, this would be addition/subtraction only.
        return F.linear(x_q, w_q, self.bias)


class BitRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (often used with BitNet)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return x_normed * self.weight
