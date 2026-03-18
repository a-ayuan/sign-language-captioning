import argparse
import time
from pathlib import Path

import pandas as pd
import torch

from slc.config import Config
from slc.datasets.wlasl import infer_feature_dim_from_manifest
from slc.inference.streaming import SlidingWindowStreamer
from slc.models.bilstm_ctc import BiLSTMCTC
from slc.preprocessing.landmarks import LandmarkExtractor
from slc.preprocessing.normalization import normalize_landmark_sequence
from slc.preprocessing.video_io import iter_video_frames
from slc.utils.io import ensure_dir, load_json
from slc.utils.metrics import compute_caption_churn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--video", type=str, required=True)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    out_dir = ensure_dir(Path(config["project"]["output_root"]) / "streaming")

    device_name = config["training"].get("device", "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    vocab = load_json(config["data"]["vocab_path"])
    index_to_token = {index: token for token, index in vocab.items()}

    configured_input_dim = int(config["data"].get("input_dim", 0))
    detected_input_dim = infer_feature_dim_from_manifest(config["data"]["train_manifest"])
    input_dim = detected_input_dim

    if configured_input_dim and configured_input_dim != detected_input_dim:
        print(
            f"Warning: config input_dim={configured_input_dim} does not match prepared features "
            f"input_dim={detected_input_dim}. Using detected_input_dim={detected_input_dim}."
        )

    model = BiLSTMCTC(
        input_dim=input_dim,
        hidden_size=int(config["model"]["hidden_size"]),
        num_layers=int(config["model"]["num_layers"]),
        vocab_size=len(vocab),
        dropout=float(config["model"]["dropout"]),
        bidirectional=bool(config["model"]["bidirectional"]),
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()

    extractor = LandmarkExtractor()
    try:
        frames = list(iter_video_frames(args.video, max_frames=int(config["data"]["max_frames"])))
        features = extractor.extract_sequence(frames)
    finally:
        extractor.close()

    features = normalize_landmark_sequence(features)

    if features.shape[1] != input_dim:
        raise ValueError(
            f"Streaming feature width mismatch: model expects input_dim={input_dim}, "
            f"but extracted video features have width={features.shape[1]}."
        )

    streamer = SlidingWindowStreamer(
        window_size=int(config["streaming"]["window_size"]),
        stride=int(config["streaming"]["stride"]),
        commit_repeats=int(config["streaming"]["commit_repeats"]),
    )

    start_time = time.perf_counter()
    results = streamer.run(
        model=model,
        features=features,
        blank_index=vocab["<blank>"],
        index_to_token=index_to_token,
        device=device,
    )
    elapsed = time.perf_counter() - start_time

    records = []
    prefixes = []
    for item in results:
        committed = " ".join(item.committed_tokens)
        prefixes.append(committed)
        records.append(
            {
                "chunk_index": item.chunk_index,
                "start_frame": item.start_frame,
                "end_frame": item.end_frame,
                "decoded_tokens": " ".join(item.decoded_tokens),
                "committed_tokens": committed,
            }
        )

    pd.DataFrame(records).to_csv(out_dir / "streaming_predictions.csv", index=False)
    summary = pd.DataFrame(
        [
            {
                "num_chunks": len(results),
                "elapsed_seconds": elapsed,
                "chunks_per_second": (len(results) / elapsed) if elapsed > 0 else 0.0,
                "caption_churn": compute_caption_churn(prefixes),
            }
        ]
    )
    summary.to_csv(out_dir / "streaming_summary.csv", index=False)


if __name__ == "__main__":
    main()