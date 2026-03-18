from dataclasses import dataclass
from typing import List

import mediapipe as mp
import numpy as np

from slc.constants import COORDS, HAND_LANDMARKS, POSE_LANDMARKS


@dataclass
class LandmarkExtractor:
    static_image_mode: bool = False
    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    def __post_init__(self) -> None:
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=self.static_image_mode,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

    def close(self) -> None:
        self.holistic.close()

    def _extract_group(self, landmarks, expected_count: int) -> np.ndarray:
        if landmarks is None:
            return np.zeros((expected_count, COORDS), dtype=np.float32)
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32)
        if coords.shape[0] != expected_count:
            padded = np.zeros((expected_count, COORDS), dtype=np.float32)
            padded[: min(expected_count, coords.shape[0])] = coords[:expected_count]
            return padded
        return coords

    def extract_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        result = self.holistic.process(frame_rgb)
        pose = self._extract_group(result.pose_landmarks, POSE_LANDMARKS)
        left_hand = self._extract_group(result.left_hand_landmarks, HAND_LANDMARKS)
        right_hand = self._extract_group(result.right_hand_landmarks, HAND_LANDMARKS)
        merged = np.concatenate([pose.reshape(-1), left_hand.reshape(-1), right_hand.reshape(-1)], axis=0)
        return merged.astype(np.float32)

    def extract_sequence(self, frames_rgb: List[np.ndarray]) -> np.ndarray:
        outputs = [self.extract_frame(frame) for frame in frames_rgb]
        if not outputs:
            feature_dim = (POSE_LANDMARKS + HAND_LANDMARKS + HAND_LANDMARKS) * COORDS
            return np.zeros((0, feature_dim), dtype=np.float32)
        return np.stack(outputs, axis=0)
