import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMCTC(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        vocab_size: int,
        dropout: float = 0.3,
        bidirectional: bool = True,
        projection_size: int | None = None,
        input_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        projection_size = int(projection_size or hidden_size)
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, projection_size),
            nn.GELU(),
            nn.Dropout(input_dropout),
        )
        self.encoder = nn.LSTM(
            input_size=projection_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        direction_factor = 2 if bidirectional else 1
        encoder_dim = hidden_size * direction_factor
        self.output_norm = nn.LayerNorm(encoder_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder_dim, vocab_size)
        self.clip_head = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, vocab_size),
        )

    def encode(self, features: torch.Tensor, input_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.input_projection(self.input_norm(features))
        packed = pack_padded_sequence(
            projected,
            lengths=input_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, _ = self.encoder(packed)
        outputs, output_lengths = pad_packed_sequence(packed_outputs, batch_first=True)
        outputs = self.dropout(self.output_norm(outputs))
        return outputs, output_lengths

    def _masked_mean_pool(self, outputs: torch.Tensor, output_lengths: torch.Tensor) -> torch.Tensor:
        time = torch.arange(outputs.size(1), device=outputs.device).unsqueeze(0)
        mask = time < output_lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).to(outputs.dtype)
        summed = (outputs * mask).sum(dim=1)
        denom = output_lengths.clamp_min(1).unsqueeze(1).to(outputs.dtype)
        return summed / denom

    def forward(self, features: torch.Tensor, input_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs, output_lengths = self.encode(features, input_lengths)
        logits = self.classifier(outputs)
        log_probs = logits.log_softmax(dim=-1)
        pooled = self._masked_mean_pool(outputs, output_lengths)
        clip_logits = self.clip_head(pooled)
        return log_probs, output_lengths, clip_logits
