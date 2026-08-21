from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from saint_george.data import build_dataloaders, export_splits
    from saint_george.engine import EvalOutput, predict_epoch, train_one_epoch, write_eval_artifacts
    from saint_george.model import build_model
    from saint_george.utils import append_jsonl, ensure_dir, format_seconds, now_slug, resolve_workspace_path, save_json, set_seed, write_text
else:
    from .data import build_dataloaders, export_splits
    from .engine import EvalOutput, predict_epoch, train_one_epoch, write_eval_artifacts
    from .model import build_model
    from .utils import append_jsonl, ensure_dir, format_seconds, now_slug, resolve_workspace_path, save_json, set_seed, write_text


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Saint George binary classifier.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--model-width", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-samples", type=int, default=0, help="Optional smoke-test cap per split.")
    parser.add_argument("--no-progress", action="store_true", help="Disable training progress bars.")
    return parser.parse_args()


def maybe_limit_split(split, limit):
    if not limit or limit <= 0:
        return split
    return split[:limit]


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = resolve_workspace_path(args.output_dir or f"runs/sg_{now_slug()}")
    ensure_dir(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = build_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    if args.limit_samples and args.limit_samples > 0:
        # Rebuild reduced loaders for smoke testing.
        from torch.utils.data import DataLoader
        if __package__ in (None, ""):
            from saint_george.data import ImageDataset
            from saint_george.transforms import build_train_transform, build_eval_transform
        else:
            from .data import ImageDataset
            from .transforms import build_train_transform, build_eval_transform

        splits = {k: maybe_limit_split(v, args.limit_samples) for k, v in loaders["splits"].items()}
        mean, std = loaders["meta"]["mean"], loaders["meta"]["std"]
        loaders["train"] = DataLoader(ImageDataset(splits["train"], build_train_transform(args.image_size, mean, std)), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        loaders["val"] = DataLoader(ImageDataset(splits["val"], build_eval_transform(args.image_size, mean, std)), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        loaders["test"] = DataLoader(ImageDataset(splits["test"], build_eval_transform(args.image_size, mean, std)), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        loaders["splits"] = splits
        loaders["meta"]["num_train"] = len(splits["train"])
        loaders["meta"]["num_val"] = len(splits["val"])
        loaders["meta"]["num_test"] = len(splits["test"])

    export_splits(loaders["splits"], output_dir)
    save_json(output_dir / "data_stats.json", loaders["meta"])

    loaders["meta"]["model_width"] = args.model_width
    model = build_model(width=args.model_width).to(device)
    train_labels = [s.label for s in loaders["splits"]["train"]]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    metrics_log = output_dir / "metrics.jsonl"
    best_f1 = -1.0
    best_epoch = -1
    best_threshold = 0.5
    stale_epochs = 0
    start_time = time.time()

    epoch_iterator = tqdm(
        range(1, args.epochs + 1),
        desc="Overall training",
        unit="epoch",
        disable=args.no_progress,
    )
    for epoch in epoch_iterator:
        train_loss = train_one_epoch(
            model,
            loaders["train"],
            device,
            criterion,
            optimizer,
            show_progress=not args.no_progress,
            desc=f"Epoch {epoch}/{args.epochs} train",
        )
        val_raw = predict_epoch(
            model,
            loaders["val"],
            device,
            criterion,
            show_progress=not args.no_progress,
            desc=f"Epoch {epoch}/{args.epochs} val",
        )
        scheduler.step(val_raw["f1"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_raw["loss"],
            "val_accuracy": val_raw["accuracy"],
            "val_precision": val_raw["precision"],
            "val_recall": val_raw["recall"],
            "val_f1": val_raw["f1"],
            "val_roc_auc": val_raw["roc_auc"],
            "threshold": val_raw["threshold"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        append_jsonl(metrics_log, row)
        epoch_iterator.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_f1=f"{val_raw['f1']:.4f}",
            val_auc=f"{val_raw['roc_auc']:.4f}",
            best_f1=f"{best_f1:.4f}",
        )
        if val_raw["f1"] > best_f1:
            best_f1 = val_raw["f1"]
            best_epoch = epoch
            best_threshold = val_raw["threshold"]
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "best_threshold": best_threshold,
                    "meta": loaders["meta"],
                    "args": vars(args),
                },
                output_dir / "best_model.pt",
            )
        else:
            stale_epochs += 1
        print(
            f"epoch {epoch:02d} train_loss={train_loss:.4f} val_f1={val_raw['f1']:.4f} "
            f"val_auc={val_raw['roc_auc']:.4f} thr={val_raw['threshold']:.2f} lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        if stale_epochs >= args.patience:
            break

    elapsed = time.time() - start_time
    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    threshold = checkpoint.get("best_threshold", best_threshold)
    test_raw = predict_epoch(
        model,
        loaders["test"],
        device,
        criterion,
        threshold=threshold,
        show_progress=not args.no_progress,
        desc="Final test",
    )
    test_result = EvalOutput(
        loss=test_raw["loss"],
        accuracy=test_raw["accuracy"],
        precision=test_raw["precision"],
        recall=test_raw["recall"],
        f1=test_raw["f1"],
        roc_auc=test_raw["roc_auc"],
        threshold=test_raw["threshold"],
        confusion=test_raw["confusion"],
        predictions=test_raw["predictions"],
    )
    write_eval_artifacts(test_result, output_dir, prefix="test")
    save_json(output_dir / "training_summary.json", {
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "best_threshold": threshold,
        "elapsed_seconds": elapsed,
        "elapsed_pretty": format_seconds(elapsed),
        "device": str(device),
        "output_dir": str(output_dir),
    })
    report = f"""# Saint George Training Report

## Configuration
- Data dir: `{args.data_dir}`
- Image size: `{args.image_size}`
- Batch size: `{args.batch_size}`
- Epochs budget: `{args.epochs}`
- Device: `{device}`

## Data Split
- Train: {loaders['meta']['num_train']}
- Val: {loaders['meta']['num_val']}
- Test: {loaders['meta']['num_test']}

## Best Validation
- Epoch: {best_epoch}
- F1: {best_f1:.4f}
- Threshold: {threshold:.2f}

## Test Results
- Accuracy: {test_result.accuracy:.4f}
- Precision: {test_result.precision:.4f}
- Recall: {test_result.recall:.4f}
- F1: {test_result.f1:.4f}
- ROC AUC: {test_result.roc_auc:.4f}

## Runtime
- Total: {format_seconds(elapsed)}

## Notes
- Weighted BCE was used to soften the class imbalance.
- Best threshold was tuned on validation predictions.
- Misclassified samples were exported under `misclassified/`.
"""
    write_text(output_dir / "report.md", report)
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
