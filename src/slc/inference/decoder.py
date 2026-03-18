from typing import Dict, List

import torch


def greedy_ctc_decode(log_probs: torch.Tensor, blank_index: int, index_to_token: Dict[int, str]) -> List[str]:
    best_path = log_probs.argmax(dim=-1).detach().cpu().tolist()
    decoded: List[str] = []
    previous = None
    for token_idx in best_path:
        if token_idx == blank_index:
            previous = token_idx
            continue
        if token_idx == previous:
            continue
        decoded.append(index_to_token[token_idx])
        previous = token_idx
    return decoded
