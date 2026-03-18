from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from slc.inference.decoder import greedy_ctc_decode
from slc.utils.metrics import compute_edit_distance_rate, compute_exact_match_accuracy
from slc.utils.visualization import save_confusion_matrix, save_training_curves


@dataclass
class EpochResult:
    loss: float
    exact_match: float
    token_error_rate: float


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
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.blank_index = blank_index
        self.index_to_token = index_to_token
        self.output_dir = Path(output_dir)
        self.grad_clip_norm = grad_clip_norm
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    def run_epoch(self, loader: DataLoader, train: bool) -> tuple[EpochResult, pd.DataFrame]:
        self.model.train(train)
        losses: List[float] = []
        all_truths: List[str] = []
        all_preds: List[str] = []
        records: List[Dict[str, str | float]] = []

        for batch in tqdm(loader, disable=False):
            features = batch["features"].to(self.device)
            targets = batch["targets"].to(self.device)
            input_lengths = batch["input_lengths"].to(self.device)
            target_lengths = batch["target_lengths"].to(self.device)

            if train:
                self.optimizer.zero_grad()

            log_probs, output_lengths = self.model(features, input_lengths)
            loss = self.criterion(
                log_probs.transpose(0, 1),
                targets,
                output_lengths,
                target_lengths,
            )

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

            losses.append(float(loss.item()))

            for index in range(features.shape[0]):
                prediction = greedy_ctc_decode(
                    log_probs[index, : output_lengths[index]],
                    blank_index=self.blank_index,
                    index_to_token=self.index_to_token,
                )
                pred_text = " ".join(prediction)
                truth_text = batch["label_texts"][index]
                all_truths.append(truth_text)
                all_preds.append(pred_text)
                records.append({"truth": truth_text, "prediction": pred_text, "path": batch["feature_paths"][index]})

        exact_match = compute_exact_match_accuracy(all_truths, all_preds)
        token_error_rate = compute_edit_distance_rate(all_truths, all_preds)
        predictions_df = pd.DataFrame(records)
        return EpochResult(np.mean(losses), exact_match, token_error_rate), predictions_df

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        early_stopping_patience: int,
    ) -> Path:
        history: List[Dict[str, float | int]] = []
        best_score = -float("inf")
        best_path = self.output_dir / "checkpoints" / "best.pt"
        patience_left = early_stopping_patience

        for epoch in range(1, epochs + 1):
            train_result, _ = self.run_epoch(train_loader, train=True)
            val_result, val_predictions = self.run_epoch(val_loader, train=False)

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_result.loss,
                    "train_exact_match": train_result.exact_match,
                    "train_token_error_rate": train_result.token_error_rate,
                    "val_loss": val_result.loss,
                    "val_exact_match": val_result.exact_match,
                    "val_token_error_rate": val_result.token_error_rate,
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

            score = val_result.exact_match - val_result.token_error_rate
            if score > best_score:
                best_score = score
                patience_left = early_stopping_patience
                torch.save({"model_state": self.model.state_dict()}, best_path)
                val_predictions.to_csv(self.output_dir / "val_predictions_best.csv", index=False)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        return best_path
