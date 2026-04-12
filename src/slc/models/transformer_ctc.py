from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerCTC(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        vocab_size: int,
        dropout: float = 0.1,
        max_len: int = 128,
        dim_feedforward: int | None = None,
        input_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.input_dropout = nn.Dropout(input_dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=int(dim_feedforward or d_model * 4),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        try:
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
                enable_nested_tensor=False,
            )
        except TypeError:
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, vocab_size)
        self.clip_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, vocab_size),
        )

    def _create_padding_mask(self, input_lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        batch_size = input_lengths.size(0)
        return torch.arange(max_len, device=input_lengths.device).expand(batch_size, max_len) >= input_lengths.unsqueeze(1)

    def encode(self, features: torch.Tensor, input_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.input_projection(self.input_norm(features))
        x = self.input_dropout(self.pos_encoder(x))
        src_key_padding_mask = self._create_padding_mask(input_lengths, features.size(1))
        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        output = self.dropout(self.output_norm(output))
        return output, input_lengths, src_key_padding_mask

    def _masked_mean_pool(self, outputs: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        valid_mask = (~src_key_padding_mask).unsqueeze(-1).to(outputs.dtype)
        summed = (outputs * valid_mask).sum(dim=1)
        denom = valid_mask.sum(dim=1).clamp_min(1.0)
        return summed / denom

    def forward(self, features: torch.Tensor, input_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output, output_lengths, src_key_padding_mask = self.encode(features, input_lengths)
        logits = self.classifier(output)
        log_probs = logits.log_softmax(dim=-1)
        pooled = self._masked_mean_pool(output, src_key_padding_mask)
        clip_logits = self.clip_head(pooled)
        return log_probs, output_lengths, clip_logits
