# models/transformer_tts.py

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerTTS(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.proj = nn.Linear(d_model, 80)  # mel bins

    def forward(self, tokens):
        # tokens: (B, T)
        x = self.embedding(tokens)
        x = self.pos_enc(x)
        x = self.encoder(x)
        mel = self.proj(x)  # (B, T, 80)
        return mel
