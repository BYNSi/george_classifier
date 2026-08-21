from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import torch
from torch import nn
from tqdm.auto import tqdm

from .utils import append_jsonl, copy_file, confusion_matrix_png, ensure_dir, save_csv, save_json


@dataclass
class EvalOutput:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    threshold: float
    confusion: list[list[int]]
    predictions: list[dict]


def select_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        preds = (probs >= threshold).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


@torch.no_grad()
def predict_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    criterion,
    threshold: float | None = None,
    show_progress: bool = True,
    desc: str = "Validation",
):
    model.eval()
    losses = []
    all_probs = []
    all_targets = []
    all_paths = []
    iterator = tqdm(
        loader,
        desc=desc,
        leave=False,
        unit="batch",
        disable=not show_progress,
    )
    for images, targets, paths in iterator:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        losses.append(loss.item() * len(targets))
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
        all_targets.append(targets.detach().cpu().numpy())
        all_paths.extend(list(paths))
    y_true = np.concatenate(all_targets).astype(int)
    probs = np.concatenate(all_probs)
    if threshold is None:
        threshold = select_threshold(y_true, probs)
    preds = (probs >= threshold).astype(int)
    metrics = {
        "loss": float(np.sum(losses) / len(y_true)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else 0.5,
        "threshold": float(threshold),
        "y_true": y_true,
        "probs": probs,
        "preds": preds,
        "paths": all_paths,
    }
    metrics["confusion"] = confusion_matrix(y_true, preds).tolist()
    predictions = [
        {
            "path": path,
            "label": int(label),
            "prob_positive": float(prob),
            "pred_label": int(pred),
            "correct": int(label == pred),
        }
        for path, label, prob, pred in zip(all_paths, y_true, probs, preds)
    ]
    metrics["predictions"] = predictions
    return metrics


def train_one_epoch(
    model,
    loader,
    device,
    criterion,
    optimizer,
    scheduler=None,
    max_grad_norm: float | None = 1.0,
    show_progress: bool = True,
    desc: str = "Training",
):
    model.train()
    losses = []
    iterator = tqdm(
        loader,
        desc=desc,
        leave=False,
        unit="batch",
        disable=not show_progress,
    )
    for images, targets, _ in iterator:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        losses.append(loss.item() * len(targets))
        iterator.set_postfix(loss=f"{loss.item():.4f}")
    if scheduler is not None:
        scheduler.step()
    return float(np.sum(losses) / len(loader.dataset))


def save_predictions_csv(predictions: Sequence[dict], path: str | Path) -> None:
    save_csv(
        path,
        list(predictions),
        fieldnames=["path", "label", "prob_positive", "pred_label", "correct"],
    )


def export_misclassifications(predictions: Sequence[dict], output_dir: str | Path) -> None:
    output_dir = ensure_dir(output_dir)
    rows = []
    for pred in predictions:
        if pred["correct"]:
            continue
        label = "fn" if pred["label"] == 1 else "fp"
        rows.append(pred)
        src = Path(pred["path"])
        dst = output_dir / label / src.name
        copy_file(src, dst)
    if rows:
        save_predictions_csv(rows, output_dir / "misclassified.csv")
    else:
        save_csv(output_dir / "misclassified.csv", [], fieldnames=["path", "label", "prob_positive", "pred_label", "correct"])


def write_eval_artifacts(result: EvalOutput, output_dir: str | Path, prefix: str = "test") -> None:
    output_dir = ensure_dir(output_dir)
    save_json(output_dir / f"{prefix}_metrics.json", {
        "loss": result.loss,
        "accuracy": result.accuracy,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "roc_auc": result.roc_auc,
        "threshold": result.threshold,
        "confusion": result.confusion,
    })
    save_predictions_csv(result.predictions, output_dir / f"{prefix}_predictions.csv")
    confusion_matrix_png(np.asarray(result.confusion), ["non_georges", "georges"], output_dir / f"{prefix}_confusion_matrix.png")
    export_misclassifications(result.predictions, output_dir / "misclassified")
