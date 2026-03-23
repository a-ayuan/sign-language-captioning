import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


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
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self, features: torch.Tensor, input_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Project input to d_model
        x = self.input_projection(features)
        # Add positional encoding
        x = self.pos_encoder(x)
        # Create mask for padding
        src_key_padding_mask = self._create_padding_mask(input_lengths, features.size(1))
        # Transformer expects (batch, seq, feature), but mask is (batch, seq)
        # Pass through transformer
        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        # Apply dropout
        output = self.dropout(output)
        # Classify
        logits = self.classifier(output)
        log_probs = logits.log_softmax(dim=-1)
        return log_probs, input_lengths

    def _create_padding_mask(self, input_lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """Create padding mask for Transformer: True for padding positions."""
        batch_size = input_lengths.size(0)
        mask = torch.arange(max_len, device=input_lengths.device).expand(batch_size, max_len) >= input_lengths.unsqueeze(1)
        return mask