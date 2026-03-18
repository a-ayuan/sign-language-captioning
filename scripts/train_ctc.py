import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from slc.config import Config
from slc.datasets.wlasl import (
    WLASLFeatureDataset,
    collate_wlasl_batch,
    infer_feature_dim_from_manifest,
)
from slc.models.bilstm_ctc import BiLSTMCTC
from slc.training.engine import Trainer
from slc.utils.io import ensure_dir, load_json
from slc.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    output_root = ensure_dir(config["project"]["output_root"])
    config.save_copy(output_root / "config_snapshot.yaml")
    set_seed(int(config["project"]["seed"]))

    device_name = config["training"].get("device", "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    vocab = load_json(config["data"]["vocab_path"])
    index_to_token = {index: token for token, index in vocab.items()}
    blank_index = vocab["<blank>"]

    train_manifest = config["data"]["train_manifest"]
    val_manifest = config["data"]["val_manifest"]

    train_dataset = WLASLFeatureDataset(
        manifest_path=train_manifest,
        vocab=vocab,
        max_frames=int(config["data"]["max_frames"]),
    )
    val_dataset = WLASLFeatureDataset(
        manifest_path=val_manifest,
        vocab=vocab,
        max_frames=int(config["data"]["max_frames"]),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["data"]["num_workers"]),
        collate_fn=collate_wlasl_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        collate_fn=collate_wlasl_batch,
    )

    configured_input_dim = int(config["data"].get("input_dim", 0))
    detected_input_dim = infer_feature_dim_from_manifest(train_manifest)
    input_dim = detected_input_dim

    if configured_input_dim and configured_input_dim != detected_input_dim:
        print(
            f"Warning: config input_dim={configured_input_dim} does not match prepared features "
            f"input_dim={detected_input_dim}. Using detected_input_dim={detected_input_dim}."
        )

    model = BiLSTMCTC(
        input_dim=input_dim,
        hidden_size=int(config["model"]["hidden_size"]),
        num_layers=int(config["model"]["num_layers"]),
        vocab_size=len(vocab),
        dropout=float(config["model"]["dropout"]),
        bidirectional=bool(config["model"]["bidirectional"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimizer"]["lr"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    criterion = nn.CTCLoss(blank=blank_index, zero_infinity=True)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        blank_index=blank_index,
        index_to_token=index_to_token,
        output_dir=output_root,
        grad_clip_norm=float(config["training"]["grad_clip_norm"]),
    )
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(config["training"]["epochs"]),
        early_stopping_patience=int(config["training"]["early_stopping_patience"]),
    )


if __name__ == "__main__":
    main()