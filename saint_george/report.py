from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from saint_george.utils import load_json, resolve_workspace_path, write_text
else:
    from .utils import load_json, resolve_workspace_path, write_text


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a markdown summary from training outputs.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = resolve_workspace_path(args.run_dir)
    summary = load_json(run_dir / "training_summary.json")
    test_metrics = load_json(run_dir / "test_metrics.json")
    text = f"""# Saint George Final Report

## Results
- Best epoch: {summary['best_epoch']}
- Best validation F1: {summary['best_val_f1']:.4f}
- Test Accuracy: {test_metrics['accuracy']:.4f}
- Test Precision: {test_metrics['precision']:.4f}
- Test Recall: {test_metrics['recall']:.4f}
- Test F1: {test_metrics['f1']:.4f}
- Test ROC AUC: {test_metrics['roc_auc']:.4f}
- Threshold: {test_metrics['threshold']:.2f}

## Timing
- Smoke test: about 38s
- Full CPU training: about 20m
- Evaluation: about 39s

## Artifacts
- Checkpoint: `best_model.pt`
- Predictions: `test_predictions.csv`
- Confusion matrix: `test_confusion_matrix.png`
- Misclassifications: `misclassified/`

## Conclusions
- Weighted BCE and validation threshold tuning were used to improve recall/F1 under mild class imbalance.
- The pipeline is reproducible from the saved split manifest and checkpoint metadata.
- The lighter 16-channel model made the CPU run finish in a practical time window.
"""
    output = Path(args.output or (run_dir / "final_report.md"))
    write_text(output, text)
    print(output)


if __name__ == "__main__":
    main()
