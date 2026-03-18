from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    root = Path("outputs/smoke_test")
    root.mkdir(parents=True, exist_ok=True)
    feature_path = root / "sample.npz"
    np.savez_compressed(feature_path, features=np.random.randn(24, 225).astype(np.float32))
    pd.DataFrame(
        [
            {
                "feature_path": str(feature_path),
                "label_text": "hello",
                "num_frames": 24,
                "split": "train",
            }
        ]
    ).to_csv(root / "manifest.csv", index=False)
    print("Smoke test artifacts written to outputs/smoke_test")


if __name__ == "__main__":
    main()
