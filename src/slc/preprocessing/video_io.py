from pathlib import Path
from typing import Generator, Tuple

import cv2
import numpy as np


def iter_video_frames(video_path: str | Path, max_frames: int | None = None) -> Generator[np.ndarray, None, None]:
    capture = cv2.VideoCapture(str(video_path))
    frame_count = 0
    try:
        while capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame
            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                break
    finally:
        capture.release()


def get_video_meta(video_path: str | Path) -> Tuple[int, float]:
    capture = cv2.VideoCapture(str(video_path))
    try:
        num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        return num_frames, fps
    finally:
        capture.release()
