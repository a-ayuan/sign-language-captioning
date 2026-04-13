from typing import List

import torch


def _edit_distance(tokens_a: List[str], tokens_b: List[str]) -> int:
    rows = len(tokens_a) + 1
    cols = len(tokens_b) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            if tokens_a[i - 1] == tokens_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + 1,
                )
    return dp[-1][-1]


def compute_exact_match_accuracy(truths: List[str], preds: List[str]) -> float:
    if not truths:
        return 0.0
    matches = sum(1 for truth, pred in zip(truths, preds) if truth.strip() == pred.strip())
    return matches / len(truths)


def compute_edit_distance_rate(truths: List[str], preds: List[str]) -> float:
    if not truths:
        return 0.0
    total_distance = 0
    total_tokens = 0
    for truth, pred in zip(truths, preds):
        truth_tokens = truth.split()
        pred_tokens = pred.split()
        total_distance += _edit_distance(truth_tokens, pred_tokens)
        total_tokens += max(1, len(truth_tokens))
    return total_distance / total_tokens


def compute_topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    if logits.numel() == 0:
        return 0.0
    k = min(k, logits.size(1))
    topk = logits.topk(k=k, dim=1).indices
    matches = topk.eq(targets.unsqueeze(1)).any(dim=1)
    return float(matches.float().mean().item())


def compute_caption_churn(prefixes: List[str]) -> float:
    if len(prefixes) < 2:
        return 0.0
    changes = sum(1 for left, right in zip(prefixes[:-1], prefixes[1:]) if left != right)
    return changes / (len(prefixes) - 1)
