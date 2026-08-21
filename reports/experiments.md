# Experiments Log

## Baseline Plan

- Model: custom residual CNN
- Loss: weighted BCEWithLogitsLoss
- Split: stratified 80/10/10
- Threshold: tuned on validation F1
- Input size: 128

## Reference Run

- Run dir: `runs/sg_20260820_220028/`
- Best validation F1: 0.6690 at epoch 9
- Test Accuracy: 0.6754
- Test F1: 0.6714
- Test ROC AUC: 0.7811
- Threshold: 0.35
- Runtime: about 20 minutes on CPU

## Planned Comparisons

1. Baseline augmentation only
2. Stronger augmentation with the same model
3. Input size comparison: 128 vs 160 vs 192
4. Optional training-time ablation: with/without class weighting

## Logging Convention

Each run stores:

- `metrics.jsonl`
- `training_summary.json`
- `test_metrics.json`
- `test_predictions.csv`
- `misclassified/`
- `report.md`
