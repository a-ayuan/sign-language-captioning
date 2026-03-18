from pathlib import Path
from typing import Dict, List

import cv2
import torch
from torch.utils.data import Dataset


class AlphabetImageDataset(Dataset):
    def __init__(self, root: str | Path, image_size: int = 224, max_items: int | None = None) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.class_names = sorted([path.name for path in self.root.iterdir() if path.is_dir()])
        self.class_to_index = {name: idx for idx, name in enumerate(self.class_names)}
        self.samples: List[tuple[str, int]] = []
        for class_name in self.class_names:
            for image_path in sorted((self.root / class_name).glob("*")):
                if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                self.samples.append((str(image_path), self.class_to_index[class_name]))
                if max_items is not None and len(self.samples) >= max_items:
                    return

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | int | str]:
        image_path, label = self.samples[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return {"image": image, "label": label, "path": image_path}
