import numpy as np


def normalize_landmark_sequence(features: np.ndarray) -> np.ndarray:
    if features.size == 0:
        return features

    normalized = features.copy()
    frame_means = normalized.mean(axis=1, keepdims=True)
    normalized = normalized - frame_means
    scale = np.std(normalized)
    if scale > 1e-6:
        normalized = normalized / scale

    deltas = np.zeros_like(normalized)
    if len(normalized) > 1:
        deltas[1:] = normalized[1:] - normalized[:-1]

    return np.concatenate([normalized, deltas], axis=1)
