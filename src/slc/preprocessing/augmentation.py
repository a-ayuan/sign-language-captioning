from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slc.preprocessing.normalization import BASE_FEATURE_DIMS


@dataclass
class LandmarkAugmenter:
    rotation_degrees: float = 10.0
    gaussian_noise_std: float = 1e-3
    temporal_speed_min: float = 0.9
    temporal_speed_max: float = 1.1
    frame_dropout_prob: float = 0.03

    def augment_sequence(self, features: np.ndarray) -> np.ndarray:
        if features.size == 0:
            return features

        if features.ndim != 2 or features.shape[1] < BASE_FEATURE_DIMS:
            raise ValueError(
                f"Expected prepared feature matrix with at least {BASE_FEATURE_DIMS} columns, got {features.shape}."
            )

        spatial = features[:, :BASE_FEATURE_DIMS].copy()
        spatial = self._rotate_xy(spatial)
        spatial = self._time_warp(spatial)
        spatial = self._frame_dropout(spatial)
        spatial = self._gaussian_noise(spatial)
        velocity = self._temporal_delta(spatial)
        return np.concatenate([spatial, velocity], axis=1).astype(np.float32)

    def _rotate_xy(self, spatial: np.ndarray) -> np.ndarray:
        if self.rotation_degrees <= 0:
            return spatial

        angle = np.deg2rad(np.random.uniform(-self.rotation_degrees, self.rotation_degrees))
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=np.float32,
        )

        rotated = spatial.copy().reshape(spatial.shape[0], -1, 3)
        rotated[:, :, :2] = rotated[:, :, :2] @ rotation.T
        return rotated.reshape(spatial.shape[0], -1)

    def _time_warp(self, spatial: np.ndarray) -> np.ndarray:
        if spatial.shape[0] < 2 or self.temporal_speed_min == 1.0 == self.temporal_speed_max:
            return spatial

        speed = np.random.uniform(self.temporal_speed_min, self.temporal_speed_max)
        if abs(speed - 1.0) < 1e-3:
            return spatial

        original_time = np.arange(spatial.shape[0], dtype=np.float32)
        warped_time = np.linspace(0, spatial.shape[0] - 1, max(2, int(round(spatial.shape[0] / speed))), dtype=np.float32)
        warped = np.empty((warped_time.shape[0], spatial.shape[1]), dtype=np.float32)
        for feature_idx in range(spatial.shape[1]):
            warped[:, feature_idx] = np.interp(warped_time, original_time, spatial[:, feature_idx])

        target_time = np.linspace(0, warped.shape[0] - 1, spatial.shape[0], dtype=np.float32)
        restored = np.empty_like(spatial)
        for feature_idx in range(spatial.shape[1]):
            restored[:, feature_idx] = np.interp(target_time, np.arange(warped.shape[0], dtype=np.float32), warped[:, feature_idx])
        return restored

    def _frame_dropout(self, spatial: np.ndarray) -> np.ndarray:
        if self.frame_dropout_prob <= 0 or spatial.shape[0] < 3:
            return spatial

        dropped = spatial.copy()
        keep_mask = np.random.random(spatial.shape[0]) >= self.frame_dropout_prob
        keep_mask[0] = True
        keep_mask[-1] = True
        if keep_mask.all():
            return dropped

        valid_idx = np.flatnonzero(keep_mask)
        for feature_idx in range(spatial.shape[1]):
            dropped[:, feature_idx] = np.interp(
                np.arange(spatial.shape[0], dtype=np.float32),
                valid_idx.astype(np.float32),
                spatial[valid_idx, feature_idx],
            )
        return dropped

    def _gaussian_noise(self, spatial: np.ndarray) -> np.ndarray:
        if self.gaussian_noise_std <= 0:
            return spatial
        noise = np.random.normal(0.0, self.gaussian_noise_std, size=spatial.shape).astype(np.float32)
        return spatial + noise

    def _temporal_delta(self, spatial: np.ndarray) -> np.ndarray:
        deltas = np.zeros_like(spatial)
        if spatial.shape[0] > 1:
            deltas[1:] = spatial[1:] - spatial[:-1]
        return deltas
