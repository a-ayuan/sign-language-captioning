from __future__ import annotations

from typing import Any, Dict

from slc.models.bilstm_ctc import BiLSTMCTC
from slc.models.transformer_ctc import TransformerCTC


def build_model(model_config: Dict[str, Any], input_dim: int, vocab_size: int, max_len: int) -> object:
    model_type = str(model_config.get("type", "bilstm")).lower()

    if model_type == "bilstm":
        return BiLSTMCTC(
            input_dim=input_dim,
            hidden_size=int(model_config["hidden_size"]),
            num_layers=int(model_config["num_layers"]),
            vocab_size=vocab_size,
            dropout=float(model_config.get("dropout", 0.3)),
            bidirectional=bool(model_config.get("bidirectional", True)),
            projection_size=int(model_config.get("projection_size", model_config["hidden_size"])),
            input_dropout=float(model_config.get("input_dropout", 0.1)),
        )

    if model_type == "transformer":
        return TransformerCTC(
            input_dim=input_dim,
            d_model=int(model_config["d_model"]),
            nhead=int(model_config["nhead"]),
            num_layers=int(model_config["num_layers"]),
            vocab_size=vocab_size,
            dropout=float(model_config.get("dropout", 0.1)),
            max_len=max_len,
            dim_feedforward=int(model_config.get("dim_feedforward", int(model_config["d_model"]) * 4)),
            input_dropout=float(model_config.get("input_dropout", 0.1)),
        )

    raise ValueError(f"Unsupported model.type: {model_type}")
