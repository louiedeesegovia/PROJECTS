# EfficientNet-V2-S Implementation for SAMP

This is a modified version of the SAMP (Saliency-Augmented Multi-Pattern Pooling) network using **EfficientNet-V2-S** as the backbone instead of ResNet.

## Changes Made

### 1. **Backbone Change**
- **Original**: ResNet-18/34/50/101 (512 or 2048 channels)
- **Modified**: EfficientNet-V2-S (1280 channels)
- The backbone function `build_efficientnet_v2_s()` extracts features using torchvision's pretrained EfficientNet-V2-S

### 2. **Configuration Updates** (config_efficientnet.py)
- `batch_size`: 16 → **4**
- `num_workers`: 8 → **0**
- `max_epoch`: **50** (unchanged)
- `lr`: **1e-4** (unchanged)
- `backbone`: Changed from `resnet_layers` to `'efficientnet_v2_s'`
- Experiment naming: Uses `'efficientnet_v2_s'` prefix instead of `'resnet{layers}'`

### 3. **Benchmarking Features Added** (test_efficientnet.py)
The test script now includes comprehensive benchmarking:

#### Metrics Reported:
1. **Inference FPS** (Frames Per Second)
   - Warmup: 20 iterations
   - Measurement: 100 iterations with CUDA synchronization
   
2. **FLOPs** (Floating Point Operations)
   - Calculated using `thop` library
   - Reported in GFLOPs
   
3. **Parameters**
   - Total model parameters
   - Reported in millions (M)
   
4. **GPU Memory Usage**
   - Peak memory allocation during inference
   - Reported in MB

#### Benchmark Function:
```python
benchmark_model(model, device)
```
This function runs before the evaluation and provides a complete performance analysis.

### 4. **File Structure**
```
Modified Files:
├── samp_net_efficientnet.py      # Network with EfficientNet-V2-S backbone
├── config_efficientnet.py         # Updated configuration
├── train_efficientnet.py          # Training script
├── test_efficientnet.py           # Testing + benchmarking script
├── requirements_efficientnet.txt  # Updated dependencies
└── README_EFFICIENTNET.md         # This file

Unchanged Files (can be reused):
├── cadb_dataset.py               # Dataset loader
├── samp_module.py                # MPP and SAMPP modules
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements_efficientnet.txt
```

2. Ensure you have the CADB dataset at `/workspace/composition/CADB_Dataset`

## Usage

### Training
```bash
python train_efficientnet.py
```

This will:
- Create experiment directory under `./experiments/`
- Train for 50 epochs with batch size 4
- Save checkpoints every epoch
- Evaluate every epoch and save best model

### Testing with Benchmarking
```bash
python test_efficientnet.py
```

This will:
1. Load the model
2. Run benchmarking (FPS, FLOPs, Params, Memory)
3. Evaluate on test set (Accuracy, EMD, MSE, SRCC, LCC)

## Expected Benchmark Output

```
============================================================
BENCHMARK METRICS
============================================================
Warming up model...
Measuring inference speed...
Inference FPS: XX.XX
Calculating FLOPs and Parameters...
FLOPs: XXX,XXX,XXX (X.XX GFLOPs)
Parameters: XX,XXX,XXX (XX.XX M)
Measuring GPU memory usage...
GPU Memory (MB): XXX.XX
============================================================
```

## Key Differences: ResNet vs EfficientNet-V2-S

| Feature | ResNet-18 | EfficientNet-V2-S |
|---------|-----------|-------------------|
| Output Channels | 512 | 1280 |
| Parameters | ~11M | ~22M |
| FLOPs | ~1.8G | ~3.0G |
| Accuracy | Baseline | Expected: Higher |
| Speed | Faster | Moderate |

## Architecture Overview

```
Input (3x224x224)
    ↓
EfficientNet-V2-S Backbone
    ↓
Feature Maps (1280 channels)
    ↓
Saliency Map Processing (MaxPool)
    ↓
SAMP Module (Multi-Pattern Pooling with Saliency)
    ↓
Attribute Branch (Optional)
    ↓
Composition Score Prediction (5 levels)
```

## Notes

- **Batch Size**: Reduced to 4 due to larger model size
- **Num Workers**: Set to 0 for debugging; increase for faster data loading
- **Pretrained Weights**: Uses ImageNet-pretrained EfficientNet-V2-S by default
- **Compatibility**: All original features (SAMP, attributes, weighted loss) are preserved

## Citation

If you use this modified implementation, please cite both the original SAMP paper and EfficientNet-V2:

```bibtex
@article{efficientnetv2,
  title={EfficientNetV2: Smaller Models and Faster Training},
  author={Tan, Mingxing and Le, Quoc},
  journal={ICML},
  year={2021}
}
```

## Troubleshooting

1. **Out of Memory**: Reduce `batch_size` in config
2. **thop not found**: Install with `pip install thop`
3. **Dataset not found**: Update `dataset_path` in config_efficientnet.py
4. **Slow training**: Increase `num_workers` in config
