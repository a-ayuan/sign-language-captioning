from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from slc.preprocessing.augmentation import LandmarkAugmenter


@dataclass
class SampleRecord:
    feature_path: str
    label_text: str
    num_frames: int
    split: str


class WLASLFeatureDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        vocab: Dict[str, int],
        max_frames: int,
        augmenter: LandmarkAugmenter | None = None,
    ) -> None:
        self.manifest = pd.read_csv(manifest_path)
        self.vocab = vocab
        self.max_frames = max_frames
        self.augmenter = augmenter

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str | int]:
        row = self.manifest.iloc[index]
        data = np.load(row["feature_path"])
        features = data["features"].astype(np.float32)
        features = features[: self.max_frames]

        if self.augmenter is not None:
            features = self.augmenter.augment_sequence(features)

        input_length = features.shape[0]
        label_tokens = [self.vocab[token] for token in row["label_text"].split()]
        if not label_tokens:
            raise ValueError(f"Encountered empty label_text at index={index} in manifest={row}")
        target = np.asarray(label_tokens, dtype=np.int64)
        class_target = int(target[0])

        return {
            "features": torch.from_numpy(features),
            "targets": torch.from_numpy(target),
            "class_target": class_target,
            "input_length": input_length,
            "target_length": len(target),
            "label_text": row["label_text"],
            "feature_path": row["feature_path"],
        }


def collate_wlasl_batch(batch: List[Dict[str, torch.Tensor | str | int]]) -> Dict[str, torch.Tensor | List[str]]:
    feature_tensors = [item["features"] for item in batch]
    input_lengths = torch.tensor([int(item["input_length"]) for item in batch], dtype=torch.long)
    target_lengths = torch.tensor([int(item["target_length"]) for item in batch], dtype=torch.long)
    class_targets = torch.tensor([int(item["class_target"]) for item in batch], dtype=torch.long)
    labels = [str(item["label_text"]) for item in batch]
    paths = [str(item["feature_path"]) for item in batch]

    max_len = max(t.shape[0] for t in feature_tensors)
    feat_dim = feature_tensors[0].shape[1]
    padded = torch.zeros((len(batch), max_len, feat_dim), dtype=torch.float32)
    for idx, tensor in enumerate(feature_tensors):
        padded[idx, : tensor.shape[0], :] = tensor

    targets = torch.cat([item["targets"] for item in batch], dim=0)

    return {
        "features": padded,
        "targets": targets,
        "class_targets": class_targets,
        "input_lengths": input_lengths,
        "target_lengths": target_lengths,
        "label_texts": labels,
        "feature_paths": paths,
    }


def infer_feature_dim_from_manifest(manifest_path: str | Path) -> int:
    manifest = pd.read_csv(manifest_path)
    if manifest.empty:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    feature_path = Path(manifest.iloc[0]["feature_path"])
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file listed in manifest does not exist: {feature_path}")

    data = np.load(feature_path)
    features = data["features"]
    if features.ndim != 2:
        raise ValueError(f"Expected 2D feature array, got shape {features.shape} from: {feature_path}")

    return int(features.shape[1])
