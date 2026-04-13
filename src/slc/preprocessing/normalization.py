import numpy as np

from slc.constants import COORDS, HAND_LANDMARKS, POSE_LANDMARKS

POSE_DIMS = POSE_LANDMARKS * COORDS
HAND_DIMS = HAND_LANDMARKS * COORDS
BASE_FEATURE_DIMS = (POSE_LANDMARKS + 2 * HAND_LANDMARKS) * COORDS

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_WRIST = 15
RIGHT_WRIST = 16


def _reshape_groups(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if features.ndim != 2 or features.shape[1] != BASE_FEATURE_DIMS:
        raise ValueError(
            f"Expected raw landmark features with shape (T, {BASE_FEATURE_DIMS}), got {features.shape}."
        )

    pose = features[:, :POSE_DIMS].reshape(-1, POSE_LANDMARKS, COORDS)
    left_hand = features[:, POSE_DIMS : POSE_DIMS + HAND_DIMS].reshape(-1, HAND_LANDMARKS, COORDS)
    right_hand = features[:, POSE_DIMS + HAND_DIMS :].reshape(-1, HAND_LANDMARKS, COORDS)
    return pose, left_hand, right_hand


def _frame_validity(group: np.ndarray) -> np.ndarray:
    return np.any(np.abs(group) > 1e-8, axis=(1, 2))


def _interpolate_group(group: np.ndarray) -> np.ndarray:
    if group.size == 0:
        return group

    valid = _frame_validity(group)
    if valid.all() or valid.sum() == 0:
        return group.copy()

    output = group.copy()
    time = np.arange(group.shape[0], dtype=np.float32)
    valid_time = time[valid]

    for joint_idx in range(group.shape[1]):
        for coord_idx in range(group.shape[2]):
            values = group[valid, joint_idx, coord_idx]
            output[:, joint_idx, coord_idx] = np.interp(time, valid_time, values)

    return output


def _safe_scale(values: np.ndarray, fallback: float = 1.0) -> np.ndarray:
    scale = values.astype(np.float32)
    scale = np.where(scale > 1e-5, scale, fallback)
    return scale


def _normalize_pose(pose: np.ndarray) -> np.ndarray:
    shoulders = pose[:, [LEFT_SHOULDER, RIGHT_SHOULDER], :]
    shoulder_center = shoulders.mean(axis=1, keepdims=True)
    shoulder_span = np.linalg.norm(
        pose[:, LEFT_SHOULDER, :2] - pose[:, RIGHT_SHOULDER, :2],
        axis=1,
        keepdims=True,
    )
    shoulder_span = _safe_scale(shoulder_span, fallback=1.0)
    return (pose - shoulder_center) / shoulder_span[:, None, :]


def _normalize_hand_local(hand: np.ndarray) -> np.ndarray:
    wrist = hand[:, :1, :]
    relative = hand - wrist
    hand_scale = np.linalg.norm(relative[:, 1:, :2], axis=2).max(axis=1, keepdims=True)
    hand_scale = _safe_scale(hand_scale, fallback=1.0)
    return relative / hand_scale[:, None, :]


def _smooth_sequence(features: np.ndarray, window_size: int = 3) -> np.ndarray:
    if features.shape[0] < 3 or window_size <= 1:
        return features

    pad = window_size // 2
    padded = np.pad(features, ((pad, pad), (0, 0)), mode="edge")
    smoothed = np.empty_like(features)
    for time_idx in range(features.shape[0]):
        smoothed[time_idx] = padded[time_idx : time_idx + window_size].mean(axis=0)
    return smoothed


def _temporal_delta(features: np.ndarray) -> np.ndarray:
    deltas = np.zeros_like(features)
    if features.shape[0] > 1:
        deltas[1:] = features[1:] - features[:-1]
    return deltas


def normalize_landmark_sequence(features: np.ndarray) -> np.ndarray:
    """Normalize landmarks for signer-invariant temporal modeling.

    The output intentionally preserves the repository's original 450-D layout:
    225 normalized spatial features + 225 first-order temporal deltas.
    """
    if features.size == 0:
        return features.astype(np.float32)

    pose, left_hand, right_hand = _reshape_groups(features.astype(np.float32))

    left_hand = _interpolate_group(left_hand)
    right_hand = _interpolate_group(right_hand)

    pose_normalized = _normalize_pose(pose)
    left_hand_normalized = _normalize_hand_local(left_hand)
    right_hand_normalized = _normalize_hand_local(right_hand)

    spatial = np.concatenate(
        [
            pose_normalized.reshape(features.shape[0], -1),
            left_hand_normalized.reshape(features.shape[0], -1),
            right_hand_normalized.reshape(features.shape[0], -1),
        ],
        axis=1,
    )
    spatial = _smooth_sequence(spatial, window_size=3)
    velocity = _temporal_delta(spatial)

    output = np.concatenate([spatial, velocity], axis=1)
    return output.astype(np.float32)
