import argparse
from pathlib import Path

import pandas as pd


def build_manifest(root: Path, output_csv: Path) -> None:
    rows = []
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for image_path in sorted(class_dir.glob("**/*")):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            rows.append({"image_path": str(image_path), "label": class_dir.name})
    pd.DataFrame(rows).to_csv(output_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument("--output-csv", type=str, required=True)
    args = parser.parse_args()
    build_manifest(Path(args.input_root), Path(args.output_csv))


if __name__ == "__main__":
    main()
