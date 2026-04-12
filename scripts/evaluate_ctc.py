import argparse
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from slc.config import Config
from slc.datasets.wlasl import WLASLFeatureDataset, collate_wlasl_batch, infer_feature_dim_from_manifest
from slc.models.factory import build_model
from slc.training.engine import Trainer
from slc.utils.io import ensure_dir, load_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    output_root = ensure_dir(Path(config["project"]["output_root"]) / "evaluation")

    device_name = config["training"].get("device", "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    vocab = load_json(config["data"]["vocab_path"])
    index_to_token = {index: token for token, index in vocab.items()}
    blank_index = vocab["<blank>"]

    test_manifest = config["data"]["test_manifest"]
    test_dataset = WLASLFeatureDataset(
        manifest_path=test_manifest,
        vocab=vocab,
        max_frames=int(config["data"]["max_frames"]),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=device.type == "cuda",
        collate_fn=collate_wlasl_batch,
    )

    configured_input_dim = int(config["data"].get("input_dim", 0))
    detected_input_dim = infer_feature_dim_from_manifest(test_manifest)
    input_dim = detected_input_dim

    if configured_input_dim and configured_input_dim != detected_input_dim:
        print(
            f"Warning: config input_dim={configured_input_dim} does not match prepared features "
            f"input_dim={detected_input_dim}. Using detected_input_dim={detected_input_dim}."
        )

    model = build_model(
        model_config=config["model"],
        input_dim=input_dim,
        vocab_size=len(vocab),
        max_len=int(config["data"]["max_frames"]),
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CTCLoss(blank=blank_index, zero_infinity=True)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        blank_index=blank_index,
        index_to_token=index_to_token,
        output_dir=output_root,
        ctc_weight=float(config.raw.get("loss", {}).get("ctc_weight", 0.2)),
        ce_weight=float(config.raw.get("loss", {}).get("ce_weight", 1.0)),
        label_smoothing=float(config.raw.get("loss", {}).get("label_smoothing", 0.0)),
        prediction_mode=str(config.raw.get("task", {}).get("prediction_mode", "clip")),
    )
    result, predictions = trainer.run_epoch(test_loader, train=False)
    pd.DataFrame(
        [
            {
                "test_loss": result.loss,
                "test_exact_match": result.exact_match,
                "test_token_error_rate": result.token_error_rate,
                "test_top5_accuracy": result.top5_accuracy,
            }
        ]
    ).to_csv(output_root / "test_metrics.csv", index=False)
    predictions.to_csv(output_root / "test_predictions.csv", index=False)


if __name__ == "__main__":
    main()
