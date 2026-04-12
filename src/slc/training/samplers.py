from __future__ import annotations

import torch
from torch.utils.data import WeightedRandomSampler

from slc.datasets.wlasl import WLASLFeatureDataset


def build_weighted_sampler(dataset: WLASLFeatureDataset) -> WeightedRandomSampler:
    label_counts = dataset.manifest["label_text"].value_counts().to_dict()
    sample_weights = dataset.manifest["label_text"].map(lambda label: 1.0 / label_counts[label]).astype(float).tolist()
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
