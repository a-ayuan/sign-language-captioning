import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from slc.config import Config
from slc.datasets.wlasl import WLASLFeatureDataset, collate_wlasl_batch, infer_feature_dim_from_manifest
from slc.models.factory import build_model
from slc.preprocessing.augmentation import LandmarkAugmenter
from slc.training.engine import Trainer
from slc.training.samplers import build_weighted_sampler
from slc.utils.io import ensure_dir, load_json
from slc.utils.seed import set_seed


def _build_augmenter(config: Config) -> LandmarkAugmenter | None:
    augmentation_config = config.raw.get("augmentation", {})
    if not augmentation_config.get("enabled", False):
        return None

    return LandmarkAugmenter(
        rotation_degrees=float(augmentation_config.get("rotation_degrees", 10.0)),
        gaussian_noise_std=float(augmentation_config.get("gaussian_noise_std", 1e-3)),
        temporal_speed_min=float(augmentation_config.get("temporal_speed_min", 0.9)),
        temporal_speed_max=float(augmentation_config.get("temporal_speed_max", 1.1)),
        frame_dropout_prob=float(augmentation_config.get("frame_dropout_prob", 0.03)),
    )


def _build_scheduler(optimizer: torch.optim.Optimizer, config: Config):
    scheduler_config = config.raw.get("scheduler", {})
    name = str(scheduler_config.get("name", "none")).lower()
    if name in {"", "none"}:
        return None, None
    if name == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(scheduler_config.get("factor", 0.5)),
            patience=int(scheduler_config.get("patience", 3)),
            min_lr=float(scheduler_config.get("min_lr", 1e-5)),
        )
        return scheduler, name
    if name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config["training"]["epochs"]),
            eta_min=float(scheduler_config.get("min_lr", 1e-5)),
        )
        return scheduler, name
    raise ValueError(f"Unsupported scheduler name: {name}")


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
        augmenter=_build_augmenter(config),
    )
    val_dataset = WLASLFeatureDataset(
        manifest_path=val_manifest,
        vocab=vocab,
        max_frames=int(config["data"]["max_frames"]),
    )

    sampler = build_weighted_sampler(train_dataset) if config["data"].get("use_weighted_sampler", False) else None
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=pin_memory,
        collate_fn=collate_wlasl_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=pin_memory,
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

    model = build_model(
        model_config=config["model"],
        input_dim=input_dim,
        vocab_size=len(vocab),
        max_len=int(config["data"]["max_frames"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimizer"]["lr"]),
        weight_decay=float(config["optimizer"].get("weight_decay", 0.0)),
    )
    scheduler, scheduler_name = _build_scheduler(optimizer, config)
    criterion = nn.CTCLoss(blank=blank_index, zero_infinity=True)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        blank_index=blank_index,
        index_to_token=index_to_token,
        output_dir=output_root,
        grad_clip_norm=float(config["training"].get("grad_clip_norm", 1.0)),
        scheduler=scheduler,
        scheduler_name=scheduler_name,
        ctc_weight=float(config.raw.get("loss", {}).get("ctc_weight", 0.2)),
        ce_weight=float(config.raw.get("loss", {}).get("ce_weight", 1.0)),
        label_smoothing=float(config.raw.get("loss", {}).get("label_smoothing", 0.0)),
        prediction_mode=str(config.raw.get("task", {}).get("prediction_mode", "clip")),
    )
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(config["training"]["epochs"]),
        early_stopping_patience=int(config["training"]["early_stopping_patience"]),
        checkpoint_metadata={
            "model_type": str(config["model"].get("type", "bilstm")),
            "input_dim": input_dim,
            "vocab_size": len(vocab),
        },
    )


if __name__ == "__main__":
    main()
