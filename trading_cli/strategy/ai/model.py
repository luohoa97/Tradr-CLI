"""AI Fusion Model — BitNet-Transformer for hybrid trading signal generation."""

import torch
import torch.nn as nn
from .bitlinear import BitLinear, BitRMSNorm


class BitNetAttention(nn.Module):
    """Multi-head Attention with ternary-quantized projections."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.q_proj = BitLinear(d_model, d_model)
        self.k_proj = BitLinear(d_model, d_model)
        self.v_proj = BitLinear(d_model, d_model)
        self.out_proj = BitLinear(d_model, d_model)
        
    def forward(self, x):
        B, T, C = x.shape
        # Ternary projections
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (self.d_head ** -0.5)
        attn = torch.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class BitNetTransformerLayer(nn.Module):
    """Single Transformer Encoder layer with BitNet components."""
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.norm1 = BitRMSNorm(d_model)
        self.attn = BitNetAttention(d_model, n_heads)
        self.norm2 = BitRMSNorm(d_model)
        self.ffn = nn.Sequential(
            BitLinear(d_model, d_ff),
            nn.SiLU(),
            BitLinear(d_ff, d_model)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class BitNetTransformer(nn.Module):
    """
    High-capacity sequence model for market pattern recognition.
    Utilizes 1.58-bit (ternary) weights for all projections.
    """
    def __init__(self, input_dim=9, d_model=512, n_heads=8, n_layers=6, seq_len=30):
        super().__init__()
        self.input_proj = BitLinear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        
        self.layers = nn.ModuleList([
            BitNetTransformerLayer(d_model, n_heads, d_model * 4)
            for _ in range(n_layers)
        ])
        
        self.norm = BitRMSNorm(d_model)
        self.head = BitLinear(d_model, 3) # 0=HOLD, 1=BUY, 2=SELL
        
    def forward(self, x):
        """
        Input x: [batch, seq_len, input_dim]
        Output: Logits [batch, 3]
        """
        # Embed and add positional information
        x = self.input_proj(x) + self.pos_embed
        
        for layer in self.layers:
            x = layer(x)
        
        # Decision based on the most recent state
        x = self.norm(x[:, -1, :])
        return self.head(x)

    @torch.no_grad()
    def predict_action(self, x):
        """Perform inference on a sequence and return discrete action."""
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        return torch.argmax(probs, dim=-1).item()


def create_model(input_dim=9, hidden_dim=512, output_dim=3, layers=6, seq_len=30):
    """Helper to instantiate the SOTA BitNet Transformer."""
    return BitNetTransformer(
        input_dim=input_dim, 
        d_model=hidden_dim, 
        n_layers=layers, 
        seq_len=seq_len
    )
