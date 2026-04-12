from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from slc.inference.decoder import greedy_ctc_decode


@dataclass
class StreamingResult:
    chunk_index: int
    start_frame: int
    end_frame: int
    decoded_tokens: List[str]
    committed_tokens: List[str]


class SlidingWindowStreamer:
    def __init__(self, window_size: int, stride: int, commit_repeats: int = 2) -> None:
        self.window_size = window_size
        self.stride = stride
        self.commit_repeats = commit_repeats
        self.recent_prefixes: List[List[str]] = []
        self.committed_tokens: List[str] = []

    def _commit_prefix(self, prefix: List[str]) -> List[str]:
        self.recent_prefixes.append(prefix)
        if len(self.recent_prefixes) < self.commit_repeats:
            return self.committed_tokens
        recent = self.recent_prefixes[-self.commit_repeats :]
        shortest = min(len(item) for item in recent)
        stable: List[str] = []
        for idx in range(shortest):
            tokens = {item[idx] for item in recent}
            if len(tokens) == 1:
                stable.append(recent[0][idx])
            else:
                break
        self.committed_tokens = stable
        return self.committed_tokens

    def run(
        self,
        model: torch.nn.Module,
        features: np.ndarray,
        blank_index: int,
        index_to_token: Dict[int, str],
        device: torch.device,
    ) -> List[StreamingResult]:
        outputs: List[StreamingResult] = []
        model.eval()
        with torch.no_grad():
            chunk_index = 0
            for start in range(0, max(1, len(features) - self.window_size + 1), self.stride):
                chunk = features[start : start + self.window_size]
                if len(chunk) == 0:
                    continue
                tensor = torch.from_numpy(chunk).unsqueeze(0).float().to(device)
                lengths = torch.tensor([tensor.shape[1]], dtype=torch.long, device=device)
                log_probs, _, clip_logits = model(tensor, lengths)
                if clip_logits is not None:
                    predicted = int(clip_logits[0].argmax(dim=-1).item())
                    decoded = [index_to_token[predicted]]
                else:
                    decoded = greedy_ctc_decode(log_probs[0], blank_index=blank_index, index_to_token=index_to_token)
                committed = list(self._commit_prefix(decoded))
                outputs.append(
                    StreamingResult(
                        chunk_index=chunk_index,
                        start_frame=start,
                        end_frame=min(len(features), start + self.window_size),
                        decoded_tokens=decoded,
                        committed_tokens=committed,
                    )
                )
                chunk_index += 1
        return outputs
