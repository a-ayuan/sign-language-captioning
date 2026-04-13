from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from fastapi import UploadFile

from slc.config import Config
from slc.datasets.wlasl import infer_feature_dim_from_manifest
from slc.inference.streaming import SlidingWindowStreamer
from slc.models.bilstm_ctc import BiLSTMCTC
from slc.preprocessing.landmarks import LandmarkExtractor
from slc.preprocessing.normalization import normalize_landmark_sequence
from slc.preprocessing.video_io import iter_video_frames
from slc.utils.io import load_json
from slc.utils.metrics import compute_caption_churn

from .schemas import ChunkPrediction, InferenceResponse


class InferenceService:
    def __init__(self, config_path: str, checkpoint_path: str) -> None:
        self.config = Config.from_yaml(config_path)

        device_name = self.config["training"].get("device", "cpu")
        if device_name == "cuda" and not torch.cuda.is_available():
            device_name = "cpu"
        self.device = torch.device(device_name)

        self.vocab = load_json(self.config["data"]["vocab_path"])
        self.index_to_token = {index: token for token, index in self.vocab.items()}
        self.blank_index = self.vocab["<blank>"]

        configured_input_dim = int(self.config["data"].get("input_dim", 0))
        detected_input_dim = infer_feature_dim_from_manifest(self.config["data"]["train_manifest"])
        self.input_dim = detected_input_dim

        if configured_input_dim and configured_input_dim != detected_input_dim:
            print(
                f"Warning: config input_dim={configured_input_dim} does not match prepared features "
                f"input_dim={detected_input_dim}. Using detected_input_dim={detected_input_dim}."
            )

        self.model = BiLSTMCTC(
            input_dim=self.input_dim,
            hidden_size=int(self.config["model"]["hidden_size"]),
            num_layers=int(self.config["model"]["num_layers"]),
            vocab_size=len(self.vocab),
            dropout=float(self.config["model"]["dropout"]),
            bidirectional=bool(self.config["model"]["bidirectional"]),
            projection_size=int(self.config["model"].get("projection_size", 0)) or None,
            input_dropout=float(self.config["model"].get("input_dropout", 0.1)),
        ).to(self.device)

        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state["model_state"])
        self.model.eval()

        self.streamer = SlidingWindowStreamer(
            window_size=int(self.config["streaming"]["window_size"]),
            stride=int(self.config["streaming"]["stride"]),
            commit_repeats=int(self.config["streaming"]["commit_repeats"]),
        )

    async def run_uploaded_video(self, upload: UploadFile) -> InferenceResponse:
        suffix = Path(upload.filename or "video.mp4").suffix or ".mp4"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            contents = await upload.read()
            temp_file.write(contents)
            temp_path = Path(temp_file.name)

        try:
            return self.run_video_path(temp_path, upload.filename or temp_path.name)
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    def run_video_path(self, video_path: Path, display_name: str | None = None) -> InferenceResponse:
        extractor = LandmarkExtractor()
        try:
            frames = list(
                iter_video_frames(
                    video_path,
                    max_frames=int(self.config["data"]["max_frames"]),
                )
            )
            if not frames:
                raise ValueError("Could not read any frames from uploaded video.")
            features = extractor.extract_sequence(frames)
        finally:
            extractor.close()

        features = normalize_landmark_sequence(features)

        if features.shape[0] == 0:
            return InferenceResponse(
                final_caption="",
                chunks=[],
                elapsed_seconds=0.0,
                num_chunks=0,
                caption_churn=0.0,
                video_filename=display_name or video_path.name,
            )

        if features.shape[1] != self.input_dim:
            raise ValueError(
                f"Streaming feature width mismatch: model expects input_dim={self.input_dim}, "
                f"but extracted video features have width={features.shape[1]}."
            )
        local_streamer = SlidingWindowStreamer(
            window_size=int(self.config["streaming"]["window_size"]),
            stride=int(self.config["streaming"]["stride"]),
            commit_repeats=int(self.config["streaming"]["commit_repeats"]),
        )

        start_time = time.perf_counter()
        results = local_streamer.run(
            model=self.model,
            features=features,
            blank_index=self.blank_index,
            index_to_token=self.index_to_token,
            device=self.device,
        )
        elapsed = time.perf_counter() - start_time

        prefixes: list[str] = []
        chunks: list[ChunkPrediction] = []

        for item in results:
            committed = " ".join(item.committed_tokens)
            prefixes.append(committed)
            chunks.append(
                ChunkPrediction(
                    chunk_index=item.chunk_index,
                    start_frame=item.start_frame,
                    end_frame=item.end_frame,
                    decoded_tokens=item.decoded_tokens,
                    committed_tokens=item.committed_tokens,
                )
            )

        final_caption = ""
        if results:
            final_caption = " ".join(results[-1].committed_tokens)

        return InferenceResponse(
            final_caption=final_caption,
            chunks=chunks,
            elapsed_seconds=elapsed,
            num_chunks=len(results),
            caption_churn=compute_caption_churn(prefixes),
            video_filename=display_name or video_path.name,
        )