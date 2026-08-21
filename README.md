# Saint George Binary Classifier

Binary image classification pipeline for detecting whether an image contains Saint George.

## Data

Expected layout:

```text
data/
  georges/
  non_georges/
```

## Train

```bash
python -m saint_george.train --data-dir data --epochs 12 --batch-size 32 --image-size 128
```

Outputs are written to `runs/sg_YYYYMMDD_HHMMSS/`.

The default CPU-friendly setup uses a 16-channel residual backbone and 128px inputs.

## Evaluate

```bash
python -m saint_george.evaluate --data-dir data --checkpoint runs/.../best_model.pt --split test
```

## Predict

```bash
python -m saint_george.predict --checkpoint runs/.../best_model.pt data/georges/sample.jpg
```

## What is included

- reproducible stratified train/val/test split
- custom PyTorch CNN
- data augmentation and normalization
- weighted BCE loss
- validation threshold tuning
- accuracy / precision / recall / F1 / ROC AUC
- confusion matrix and misclassification export

## Notes

This repository is CPU-friendly and does not depend on `torchvision` or `timm`.

The latest reference run is stored under `runs/sg_20260820_220028/`.

## Publish to GitHub

The image dataset is intentionally excluded from Git because it is about 0.54 GB.
Keep the local dataset in:

```text
data/
  georges/
  non_georges/
```

Create an empty repository on GitHub, then run these commands from the project root:

```bash
git init
git add .
git commit -m "Initial Saint George classifier pipeline"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Replace `YOUR_USERNAME` and `YOUR_REPOSITORY` with your GitHub account and repository name.
