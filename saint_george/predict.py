from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from saint_george.model import build_model
    from saint_george.transforms import build_eval_transform
    from saint_george.utils import load_json, resolve_workspace_path
else:
    from .model import build_model
    from .transforms import build_eval_transform
    from .utils import load_json, resolve_workspace_path


def parse_args():
    parser = argparse.ArgumentParser(description="Predict Saint George class for an image or directory.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("path", type=str)
    parser.add_argument("--threshold", type=float, default=None)
    return parser.parse_args()


def predict_one(model, transform, path: Path, device, threshold: float):
    with Image.open(path) as image:
        tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).item()
    label = "georges" if prob >= threshold else "non_georges"
    return {"path": str(path), "prob_positive": prob, "label": label, "threshold": threshold}


def main():
    args = parse_args()
    checkpoint = torch.load(resolve_workspace_path(args.checkpoint), map_location="cpu", weights_only=False)
    meta = checkpoint["meta"]
    threshold = args.threshold if args.threshold is not None else checkpoint.get("best_threshold", 0.5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(width=checkpoint.get("args", {}).get("model_width", meta.get("model_width", 16))).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    transform = build_eval_transform(meta["image_size"], meta["mean"], meta["std"])
    path = Path(args.path)
    if path.is_dir():
        for item in sorted(path.glob("*")):
            if item.is_file():
                result = predict_one(model, transform, item, device, threshold)
                print(result)
    else:
        print(predict_one(model, transform, path, device, threshold))


if __name__ == "__main__":
    main()
