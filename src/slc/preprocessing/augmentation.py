import numpy as np
from typing import List


class LandmarkAugmenter:
    """Data augmentation for landmark sequences."""

    def __init__(self, prob_temporal_interpolate=0.3, prob_spatial_jitter=0.2,
                 prob_temporal_flip=0.15, prob_spatial_scale=(0.8, 1.2)):
        self.prob_temporal_interpolate = prob_temporal_interpolate
        self.prob_spatial_jitter = prob_spatial_jitter
        self.prob_temporal_flip = prob_temporal_flip
        self.spatial_scale_range = prob_spatial_scale

    def augment_sequence(self, features: np.ndarray) -> np.ndarray:
        """Apply random augmentations to a landmark sequence."""
        augmented = features.copy()

        # Temporal interpolations
        if np.random.random() < self.prob_temporal_interpolate:
            augmented = self._temporal_interpolate(augmented)

        # Spatial jitter
        if np.random.random() < self.prob_spatial_jitter:
            augmented = self._spatial_jitter(augmented)

        # Temporal flip
        if np.random.random() < self.prob_temporal_flip:
            augmented = self._temporal_flip(augmented)

        # Spatial scaling
        scale_factor = np.random.uniform(self.spatial_scale_range[0], self.spatial_scale_range[1])
        augmented = self._spatial_scale(augmented, scale_factor)

        return augmented

    def _temporal_interpolate(self, features: np.ndarray) -> np.ndarray:
        """Insert interpolated frames to handle speed variation."""
        if features.shape[0] < 2:
            return features

        # Randomly insert interpolated frames
        new_frames = []
        for i in range(features.shape[0] - 1):
            new_frames.append(features[i])
            if np.random.random() < 0.5:  # 50% chance to interpolate between frames
                interpolated = (features[i] + features[i + 1]) / 2
                new_frames.append(interpolated)

        new_frames.append(features[-1])
        return np.stack(new_frames, axis=0)

    def _spatial_jitter(self, features: np.ndarray, std: float = 0.02) -> np.ndarray:
        """Add Gaussian noise to landmarks (signer variation)."""
        noise = np.random.normal(0, std, features.shape)
        return features + noise

    def _temporal_flip(self, features: np.ndarray) -> np.ndarray:
        """Mirror frames (mirrors are common in ASL videos)."""
        return features[::-1].copy()

    def _spatial_scale(self, features: np.ndarray, scale: float) -> np.ndarray:
        """Random scale (distance from camera varies)."""
        # Only scale the spatial coordinates, not additional features
        # Assuming first part is spatial landmarks
        spatial_dim = 450  # Base landmark dimensions
        scaled_features = features.copy()
        scaled_features[:, :spatial_dim] *= scale
        return scaled_features