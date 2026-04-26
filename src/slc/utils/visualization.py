from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def save_training_curves(history_df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 6))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    plt.plot(history_df["epoch"], history_df["train_exact_match"], label="train_exact_match")
    plt.plot(history_df["epoch"], history_df["val_exact_match"], label="val_exact_match")
    plt.plot(history_df["epoch"], history_df["train_token_error_rate"], label="train_token_error_rate")
    plt.plot(history_df["epoch"], history_df["val_token_error_rate"], label="val_token_error_rate")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Curves")
    plt.legend()
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_confusion_matrix(
    truths: List[str], 
    preds: List[str], 
    output_path: str | Path, 
    model_name: str = "Base BiLSTM",
    max_classes: int = 20
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not truths:
        return

    counts = pd.Series(truths).value_counts().head(max_classes)
    keep = set(counts.index.tolist())
    filtered_truths = [item for item in truths if item in keep]
    filtered_preds = [pred if pred in keep else "<other>" for truth, pred in zip(truths, preds) if truth in keep]
    labels = sorted(list(keep | {"<other>"}))

    matrix = confusion_matrix(filtered_truths, filtered_preds, labels=labels, normalize='true')
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(matrix, interpolation="nearest")
    
    plt.title(f"Validation Confusion Matrix: {model_name}")
    
    cbar = plt.colorbar()
    cbar.set_label("Proportion of True Class", rotation=270, labelpad=15)
    
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_class_distribution(labels: List[str], output_path: str | Path, top_k: int = 30) -> None:
    output_path = Path(output_path)
    counts = pd.Series(labels).value_counts().head(top_k)
    fig = plt.figure(figsize=(12, 6))
    counts.plot(kind="bar")
    plt.title("Class Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_sequence_length_histogram(lengths: List[int], output_path: str | Path) -> None:
    output_path = Path(output_path)
    fig = plt.figure(figsize=(8, 5))
    plt.hist(lengths, bins=30)
    plt.title("Sequence Length Distribution")
    plt.xlabel("Frames")
    plt.ylabel("Count")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_sample_trajectory(features: np.ndarray, output_path: str | Path) -> None:
    output_path = Path(output_path)
    if features.size == 0:
        return
    limit = min(features.shape[1], 12)
    fig = plt.figure(figsize=(10, 6))
    for idx in range(limit):
        plt.plot(features[:, idx], label=f"dim_{idx}")
    plt.title("Sample Landmark Trajectory")
    plt.xlabel("Frame")
    plt.ylabel("Normalized Value")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
