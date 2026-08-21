from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw
import torch


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_workspace_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return (repo_root() / path).resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def now_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(path: str | Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        fieldnames = fieldnames or []
    elif fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m"


def timing_block() -> tuple[float, callable]:
    start = time.time()

    def stop() -> float:
        return time.time() - start

    return start, stop


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def confusion_matrix_png(
    matrix: np.ndarray,
    labels: Sequence[str],
    path: str | Path,
    title: str = "Confusion matrix",
) -> None:
    matrix = np.asarray(matrix)
    size = 420
    pad = 48
    cell = (size - 2 * pad) // max(2, matrix.shape[0])
    width = height = size
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    max_val = max(1, int(matrix.max()))
    draw.text((pad, 12), title, fill="black")

    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            x0 = pad + j * cell
            y0 = pad + i * cell
            x1 = x0 + cell
            y1 = y0 + cell
            shade = 255 - int(180 * (val / max_val))
            fill = (255, shade, shade)
            draw.rectangle([x0, y0, x1, y1], fill=fill, outline="black")
            text = str(int(val))
            bbox = draw.textbbox((0, 0), text)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text((x0 + (cell - tw) / 2, y0 + (cell - th) / 2), text, fill="black")

    for idx, label in enumerate(labels):
        draw.text((pad + idx * cell + 8, pad - 22), label, fill="black")
        draw.text((8, pad + idx * cell + 8), label, fill="black")

    ensure_dir(Path(path).parent)
    image.save(path)


def copy_file(src: str | Path, dst: str | Path) -> None:
    src = Path(src)
    dst = Path(dst)
    ensure_dir(dst.parent)
    dst.write_bytes(src.read_bytes())


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")
