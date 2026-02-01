# models/mamba_tts.py

import torch
import torch.nn as nn


class MambaBlock(nn.Module):
    """
    Pure PyTorch Mamba-style SSM block (CPU-safe)
    """
    def __init__(self, d_model):
        super().__init__()

        self.in_proj = nn.Linear(d_model, 2 * d_model)
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=5,
            padding=4,
            groups=d_model
        )
        self.out_proj = nn.Linear(d_model, d_model)
        self.act = nn.SiLU()

    def forward(self, x):
        # x: (B, T, D)
        u, v = self.in_proj(x).chunk(2, dim=-1)

        # temporal mixing (state-space–like)
        v = v.transpose(1, 2)                # (B, D, T)
        v = self.depthwise_conv(v)[..., :u.size(1)]
        v = v.transpose(1, 2)                # (B, T, D)

        return self.out_proj(self.act(u * v))


class MambaTTS(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=4):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList(
            [MambaBlock(d_model) for _ in range(n_layers)]
        )

        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, 80)

    def forward(self, tokens):
        x = self.embedding(tokens)
        for layer in self.layers:
            x = x + layer(x)   # residual connection
        x = self.norm(x)
        return self.proj(x)
    