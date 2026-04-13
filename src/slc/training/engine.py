from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from slc.inference.decoder import greedy_ctc_decode
from slc.utils.metrics import compute_edit_distance_rate, compute_exact_match_accuracy, compute_topk_accuracy
from slc.utils.visualization import save_confusion_matrix, save_training_curves


@dataclass
class EpochResult:
    loss: float
    exact_match: float
    token_error_rate: float
    top5_accuracy: float


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        blank_index: int,
        index_to_token: Dict[int, str],
        output_dir: str | Path,
        grad_clip_norm: float = 5.0,
        scheduler: object | None = None,
        scheduler_name: str | None = None,
        ctc_weight: float = 0.2,
        ce_weight: float = 1.0,
        label_smoothing: float = 0.0,
        prediction_mode: str = "clip",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.blank_index = blank_index
        self.index_to_token = index_to_token
        self.output_dir = Path(output_dir)
        self.grad_clip_norm = grad_clip_norm
        self.scheduler = scheduler
        self.scheduler_name = (scheduler_name or "").lower()
        self.ctc_weight = float(ctc_weight)
        self.ce_weight = float(ce_weight)
        self.label_smoothing = float(label_smoothing)
        self.prediction_mode = prediction_mode
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    def _decode_predictions(
        self,
        log_probs: torch.Tensor,
        output_lengths: torch.Tensor,
        clip_logits: torch.Tensor,
        batch: Dict[str, torch.Tensor | List[str]],
        records: List[Dict[str, str | float]],
        all_truths: List[str],
        all_preds: List[str],
    ) -> tuple[List[float], List[int]]:
        batch_top5: List[float] = []
        batch_class_targets: List[int] = []
        class_targets = batch["class_targets"]
        assert isinstance(class_targets, torch.Tensor)
        for index in range(log_probs.shape[0]):
            truth_text = batch["label_texts"][index]
            clip_prediction = self.index_to_token[int(clip_logits[index].argmax(dim=-1).item())]
            ctc_prediction = " ".join(
                greedy_ctc_decode(
                    log_probs[index, : output_lengths[index]],
                    blank_index=self.blank_index,
                    index_to_token=self.index_to_token,
                )
            )
            pred_text = clip_prediction if self.prediction_mode == "clip" else ctc_prediction
            all_truths.append(truth_text)
            all_preds.append(pred_text)
            records.append(
                {
                    "truth": truth_text,
                    "prediction": pred_text,
                    "clip_prediction": clip_prediction,
                    "ctc_prediction": ctc_prediction,
                    "path": batch["feature_paths"][index],
                }
            )
            batch_class_targets.append(int(class_targets[index].item()))
        return batch_top5, batch_class_targets

    def run_epoch(self, loader: DataLoader, train: bool) -> tuple[EpochResult, pd.DataFrame]:
        self.model.train(train)
        losses: List[float] = []
        all_truths: List[str] = []
        all_preds: List[str] = []
        records: List[Dict[str, str | float]] = []
        top5_scores: List[float] = []

        for batch in tqdm(loader, disable=False):
            features = batch["features"].to(self.device)
            targets = batch["targets"].to(self.device)
            class_targets = batch["class_targets"].to(self.device)
            input_lengths = batch["input_lengths"].to(self.device)
            target_lengths = batch["target_lengths"].to(self.device)

            if train:
                self.optimizer.zero_grad(set_to_none=True)

            log_probs, output_lengths, clip_logits = self.model(features, input_lengths)

            total_loss = torch.tensor(0.0, device=self.device)
            if self.ctc_weight > 0.0:
                ctc_loss = self.criterion(
                    log_probs.transpose(0, 1),
                    targets,
                    output_lengths,
                    target_lengths,
                )
                total_loss = total_loss + (self.ctc_weight * ctc_loss)
            if self.ce_weight > 0.0:
                ce_loss = F.cross_entropy(
                    clip_logits,
                    class_targets,
                    label_smoothing=self.label_smoothing,
                )
                total_loss = total_loss + (self.ce_weight * ce_loss)

            if train:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

            losses.append(float(total_loss.item()))
            top5_scores.append(compute_topk_accuracy(clip_logits.detach(), class_targets.detach(), k=5))
            self._decode_predictions(log_probs, output_lengths, clip_logits, batch, records, all_truths, all_preds)

        exact_match = compute_exact_match_accuracy(all_truths, all_preds)
        token_error_rate = compute_edit_distance_rate(all_truths, all_preds)
        predictions_df = pd.DataFrame(records)
        return EpochResult(float(np.mean(losses)), exact_match, token_error_rate, float(np.mean(top5_scores))), predictions_df

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        early_stopping_patience: int,
        checkpoint_metadata: Dict[str, object] | None = None,
    ) -> Path:
        history: List[Dict[str, float | int]] = []
        best_score = -float("inf")
        best_path = self.output_dir / "checkpoints" / "best.pt"
        patience_left = early_stopping_patience
        checkpoint_metadata = checkpoint_metadata or {}

        for epoch in range(1, epochs + 1):
            train_result, _ = self.run_epoch(train_loader, train=True)
            val_result, val_predictions = self.run_epoch(val_loader, train=False)

            if self.scheduler is not None:
                if self.scheduler_name == "reduce_on_plateau":
                    self.scheduler.step(val_result.loss)
                else:
                    self.scheduler.step()

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_result.loss,
                    "train_exact_match": train_result.exact_match,
                    "train_token_error_rate": train_result.token_error_rate,
                    "train_top5_accuracy": train_result.top5_accuracy,
                    "val_loss": val_result.loss,
                    "val_exact_match": val_result.exact_match,
                    "val_token_error_rate": val_result.token_error_rate,
                    "val_top5_accuracy": val_result.top5_accuracy,
                }
            )

            pd.DataFrame(history).to_csv(self.output_dir / "metrics_history.csv", index=False)
            val_predictions.to_csv(self.output_dir / "val_predictions_latest.csv", index=False)
            save_training_curves(pd.DataFrame(history), self.output_dir / "training_curves.png")
            save_confusion_matrix(
                truths=val_predictions["truth"].tolist(),
                preds=val_predictions["prediction"].tolist(),
                output_path=self.output_dir / "val_confusion_matrix.png",
            )

            score = (2.0 * val_result.exact_match) + val_result.top5_accuracy - val_result.token_error_rate
            if score > best_score:
                best_score = score
                patience_left = early_stopping_patience
                torch.save(
                    {
                        "model_state": self.model.state_dict(),
                        "best_score": best_score,
                        "checkpoint_metadata": checkpoint_metadata,
                    },
                    best_path,
                )
                val_predictions.to_csv(self.output_dir / "val_predictions_best.csv", index=False)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        return best_path
