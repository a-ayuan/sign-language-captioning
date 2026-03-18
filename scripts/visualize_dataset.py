import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from slc.utils.visualization import (
    save_class_distribution,
    save_sample_trajectory,
    save_sequence_length_histogram,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_class_distribution(manifest["label_text"].tolist(), output_dir / "class_distribution.png")
    save_sequence_length_histogram(manifest["num_frames"].tolist(), output_dir / "sequence_lengths.png")

    if len(manifest) > 0:
        sample = np.load(manifest.iloc[0]["feature_path"])
        save_sample_trajectory(sample["features"], output_dir / "sample_trajectory.png")

    manifest.describe(include="all").to_csv(output_dir / "dataset_summary.csv")


if __name__ == "__main__":
    main()
