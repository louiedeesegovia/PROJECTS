# Thesis 1 — SAMP-Net Backbone Benchmarking

This folder contains six variants of **SAMP-Net** (Saliency-Aware Multi-Pattern Pooling Network), a deep learning model for **automatic image composition assessment**. Each subfolder swaps the original ResNet backbone for a different, more lightweight CNN architecture, so the results can be benchmarked against one another for speed, size, and accuracy trade-offs.

## Backbones Compared

| Folder | Backbone | Notes |
|---|---|---|
| `CspNet (Cross-Stage Partial Networks)` | CSPDarkNet53 / CSPResNet50 | Better gradient flow via cross-stage partial connections |
| `EfficientNet-B0` | EfficientNet-B0 | Compound-scaled, ImageNet-pretrained, 1280-channel output |
| `EfficientNet-V2-S` | EfficientNet-V2-S | Improved EfficientNet variant, 1280-channel output |
| `Inception-v1` | Inception-v1 (GoogLeNet) | 1024-channel output |
| `MobileNetV3-Small` | MobileNetV3-Small | Mobile-optimized, 576-channel output |
| `RegNetX-400MF` | RegNetX-400MF | ~5.2M params, designed for edge/laptop training, 400-channel output |

Each backbone folder is a self-contained, runnable copy of the pipeline (config, dataset loader, network, training, and testing/benchmarking scripts), modified only in the backbone-specific parts.

## What the Model Does

SAMP-Net predicts an **image composition quality score** (and related composition attributes) from a photo, using:
- A **CNN backbone** (varies per folder — see table above) for feature extraction
- A **Multi-Pattern Pooling module** that pools features according to common composition patterns (rule of thirds, symmetry, etc.)
- A **saliency-aware** weighting mechanism
- A **multi-task attribute branch** predicting attributes such as `RuleOfThirds`, `BalancingElements`, `DoF`, `Object`, `Symmetry`, and `Repetition`
- A **score distribution head** trained with Earth Mover's Distance (EMD) loss

Training/evaluation is done on the **CADB dataset** (Composition Assessment Database).

## Common File Structure (per backbone folder)

```
config.py            # Hyperparameters, dataset path, backbone selection
cadb_dataset.py       # CADB dataset loader
samp_net.py           # SAMP-Net architecture (backbone + pooling + heads)
samp_module.py         # Multi-pattern pooling module implementation
train.py              # Training script
test.py               # Evaluation + benchmarking script
requirements.txt       # Python dependencies
README*.md            # Backbone-specific notes on what was changed
```

Some folders include extra docs (`BENCHMARKING_GUIDE.md`, `CHANGES_SUMMARY.md`, `QUICK_START.md`, `MODIFICATION_SUMMARY.md`, etc.) with more detail on the specific modifications made for that backbone.

## Shared Configuration Defaults

All variants use a common baseline configuration, adjusted for lower-end/laptop hardware:

```python
batch_size = 4
num_workers = 0
max_epoch = 50
lr = 1e-4
optimizer = 'adam'
image_size = 224
```

## Benchmark Metrics

Each `test.py` reports both **accuracy** and **efficiency** metrics so backbones can be compared directly:

**Accuracy**
- Accuracy (binary classification threshold)
- EMD (r=1), EMD (r=2)
- MSE
- SRCC (Spearman Rank Correlation Coefficient)
- LCC (Linear Correlation Coefficient)

**Efficiency**
- Inference FPS
- Parameters (M)
- FLOPs (G) — via `thop`
- Peak GPU memory (MB)

## Setup

1. Install dependencies inside the desired backbone folder:
   ```bash
   pip install -r requirements.txt
   ```
2. Update `dataset_path` in `config.py` to point to your local copy of the CADB dataset.
3. Train:
   ```bash
   python train.py
   ```
4. Evaluate and benchmark:
   ```bash
   python test.py
   ```

## Purpose

This set of experiments is part of a thesis comparing lightweight/efficient CNN backbones as drop-in replacements for ResNet in SAMP-Net, to evaluate the accuracy vs. computational-cost trade-off for image composition assessment on resource-constrained hardware.
