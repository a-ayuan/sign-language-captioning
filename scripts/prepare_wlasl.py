import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from slc.data.kaggle_wlasl import KaggleDownloadError, prepare_kaggle_wlasl_root
from slc.preprocessing.landmarks import LandmarkExtractor
from slc.preprocessing.normalization import normalize_landmark_sequence
from slc.preprocessing.video_io import get_video_meta, iter_video_frames
from slc.utils.io import ensure_dir, save_json


def discover_split_videos(split_root: Path) -> List[tuple[str, str]]:
    samples: List[tuple[str, str]] = []
    for label_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
        for video_path in sorted(label_dir.glob("**/*")):
            if video_path.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
                continue
            samples.append((label_dir.name.lower().strip(), str(video_path)))
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=str, default="data/wlasl_processed")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--max-frames", type=int, default=96)
    parser.add_argument("--top-k", type=int, default=0, help="Use the K most frequent glosses across all splits. 0 keeps all classes.")
    parser.add_argument(
        "--dataset-slug",
        type=str,
        default="risangbaskoro/wlasl-processed",
        help="Kaggle dataset slug from the proposal.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help=(
            "Download and stage the Kaggle WLASL dataset automatically when needed. "
            "If train/val/test folders are not present in the download, the script builds them "
            "from the WLASL metadata JSON."
        ),
    )
    parser.add_argument(
        "--cache-root",
        type=str,
        default=str(Path.home() / ".cache" / "slc_kaggle"),
        help="Local cache directory for Kaggle downloads.",
    )
    return parser.parse_args()


def _select_top_k_labels(split_to_samples: Dict[str, List[tuple[str, str]]], top_k: int) -> set[str]:
    counts: Counter[str] = Counter()
    for samples in split_to_samples.values():
        for label_text, _ in samples:
            counts[label_text] += 1
    if top_k <= 0 or top_k >= len(counts):
        return set(counts.keys())
    return {label for label, _ in counts.most_common(top_k)}


def main() -> None:
    args = parse_args()

    input_root = Path(args.input_root)
    if args.download or not input_root.exists():
        try:
            input_root = prepare_kaggle_wlasl_root(
                dataset_slug=args.dataset_slug,
                target_root=input_root,
                cache_root=Path(args.cache_root),
            )
            print(f"WLASL dataset ready at: {input_root}")
        except KaggleDownloadError as exc:
            raise SystemExit(f"Dataset setup failed: {exc}") from exc

    output_root = ensure_dir(args.output_root)
    feature_root = ensure_dir(output_root / "features")
    manifest_root = ensure_dir(output_root / "manifests")

    split_to_samples: Dict[str, List[tuple[str, str]]] = {}
    for split in ["train", "val", "test"]:
        split_root = input_root / split
        split_to_samples[split] = discover_split_videos(split_root) if split_root.exists() else []

    selected_labels = _select_top_k_labels(split_to_samples, top_k=int(args.top_k))
    if args.top_k > 0:
        print(f"Keeping top-{args.top_k} glosses by frequency. Selected {len(selected_labels)} labels.")

    extractor = LandmarkExtractor()
    vocab: Dict[str, int] = {"<blank>": 0}
    summary_rows: List[Dict[str, str | int | float]] = []

    try:
        for split in ["train", "val", "test"]:
            rows = []
            split_samples = split_to_samples[split]
            for label_text, video_path_str in tqdm(split_samples, desc=f"prepare_{split}"):
                if label_text not in selected_labels:
                    continue
                if label_text not in vocab:
                    vocab[label_text] = len(vocab)

                video_path = Path(video_path_str)
                frames = list(iter_video_frames(video_path, max_frames=args.max_frames))
                raw_features = extractor.extract_sequence(frames)
                normalized = normalize_landmark_sequence(raw_features)
                if normalized.shape[0] == 0:
                    continue

                out_name = f"{split}_{label_text}_{video_path.stem}.npz"
                feature_path = feature_root / out_name
                feature_path.parent.mkdir(parents=True, exist_ok=True)

                num_frames_meta, fps = get_video_meta(video_path)
                np.savez_compressed(feature_path, features=normalized)

                rows.append(
                    {
                        "feature_path": str(feature_path),
                        "label_text": label_text,
                        "num_frames": int(normalized.shape[0]),
                        "split": split,
                    }
                )
                summary_rows.append(
                    {
                        "split": split,
                        "label_text": label_text,
                        "video_path": str(video_path),
                        "feature_path": str(feature_path),
                        "raw_num_frames": num_frames_meta,
                        "used_num_frames": int(normalized.shape[0]),
                        "fps": fps,
                    }
                )

            pd.DataFrame(rows).to_csv(manifest_root / f"{split}.csv", index=False)
    finally:
        extractor.close()

    save_json(vocab, manifest_root / "vocab.json")
    pd.DataFrame(summary_rows).to_csv(output_root / "preprocessing_summary.csv", index=False)


if __name__ == "__main__":
    main()
