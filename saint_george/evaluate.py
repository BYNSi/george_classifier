from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch import nn

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from saint_george.data import build_dataloaders
    from saint_george.engine import EvalOutput, predict_epoch, write_eval_artifacts
    from saint_george.model import build_model
    from saint_george.utils import load_json, resolve_workspace_path, save_json, set_seed
else:
    from .data import build_dataloaders
    from .engine import EvalOutput, predict_epoch, write_eval_artifacts
    from .model import build_model
    from .utils import load_json, resolve_workspace_path, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a Saint George checkpoint.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    meta = checkpoint["meta"]
    output_dir = resolve_workspace_path(args.output_dir or Path(args.checkpoint).parent)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = build_dataloaders(
        data_dir=args.data_dir,
        image_size=meta["image_size"],
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    model = build_model(width=checkpoint.get("args", {}).get("model_width", meta.get("model_width", 16))).to(device)
    model.load_state_dict(checkpoint["model_state"])
    criterion = nn.BCEWithLogitsLoss()
    raw = predict_epoch(model, loaders[args.split], device, criterion, threshold=checkpoint.get("best_threshold", 0.5))
    result = EvalOutput(
        loss=raw["loss"],
        accuracy=raw["accuracy"],
        precision=raw["precision"],
        recall=raw["recall"],
        f1=raw["f1"],
        roc_auc=raw["roc_auc"],
        threshold=raw["threshold"],
        confusion=raw["confusion"],
        predictions=raw["predictions"],
    )
    write_eval_artifacts(result, output_dir, prefix=args.split)
    save_json(output_dir / f"{args.split}_summary.json", {
        "split": args.split,
        "loss": result.loss,
        "accuracy": result.accuracy,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "roc_auc": result.roc_auc,
        "threshold": result.threshold,
    })
    print(f"{args.split} metrics saved to {output_dir}")


if __name__ == "__main__":
    main()
