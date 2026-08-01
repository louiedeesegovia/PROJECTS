# EfficientNet-B0 SAMP Implementation for Benchmarking

## Overview
This is a modified version of the SAMP (Saliency-Aware Multi-Pattern) network using **EfficientNet-B0** as the backbone instead of ResNet. This implementation is designed for benchmarking purposes with comprehensive performance metrics.

## Key Changes

### 1. Backbone Architecture
- **Original**: ResNet-18/34/50/101
- **Modified**: EfficientNet-B0
- **Output channels**: 1280 (EfficientNet-B0) vs 512/2048 (ResNet)
- **Spatial resolution**: 7×7 feature maps

### 2. Configuration Changes (config.py)
```python
batch_size = 4          # Changed from 16
num_workers = 0         # Changed from 8
lr = 1e-4              # Maintained
max_epoch = 50         # Maintained
backbone_type = 'efficientnet_b0'  # New parameter
```

### 3. Network Architecture (samp_net.py)
- Added `build_efficientnet_b0()` function
- Modified `SAMPNet.__init__()` to support both ResNet and EfficientNet-B0
- Uses `torchvision.models.efficientnet_b0` with ImageNet pretrained weights
- Automatically adjusts input channels based on backbone type

### 4. Benchmark Metrics (test.py)
The evaluation script now reports comprehensive benchmark metrics:

#### Performance Metrics:
- **Inference FPS**: Frames per second on GPU
- **Parameters (M)**: Total number of model parameters in millions
- **FLOPs (G)**: Floating point operations in billions
- **GPU Memory (MB)**: Peak GPU memory usage

#### Accuracy Metrics:
- **Accuracy**: Binary classification accuracy (threshold=2.6)
- **EMD (r=1)**: Earth Mover's Distance with r=1
- **EMD (r=2)**: Earth Mover's Distance with r=2
- **MSE**: Mean Squared Error
- **SRCC**: Spearman Rank Correlation Coefficient
- **LCC**: Linear Correlation Coefficient

### 5. Dependencies
Added `thop` package for FLOPs calculation:
```bash
pip install thop
```

## Usage

### Training
```bash
python train.py
```

### Testing with Benchmarking
```bash
python test.py
```

The test script will automatically:
1. Warm up the model (20 iterations)
2. Measure inference speed (100 iterations)
3. Calculate FLOPs and parameters
4. Measure GPU memory usage
5. Evaluate on the test dataset
6. Print comprehensive summary

## Expected Output

```
==============================================================
BENCHMARK METRICS
==============================================================

Warming up...
Measuring inference speed...
Calculating FLOPs and Parameters...
Measuring GPU memory usage...

------------------------------------------------------------
BENCHMARK RESULTS:
------------------------------------------------------------
Inference FPS:        XX.XX
Parameters (M):       XX.XX
FLOPs (G):            XX.XX
GPU Memory (MB):      XX.XX
------------------------------------------------------------

Evaluation begining...
[Progress bar...]
Evaluation result...
Test on XXX images, Accuracy=XX.XX%, EMD(r=1)=X.XXXX, EMD(r=2)=X.XXXX, MSE_loss=X.XXXX, SRCC=X.XXXX, LCC=X.XXXX

==============================================================
FINAL SUMMARY
==============================================================
Accuracy:             XX.XX%
EMD (r=1):            X.XXXX
EMD (r=2):            X.XXXX
MSE:                  X.XXXX
SRCC:                 X.XXXX
LCC:                  X.XXXX
Inference FPS:        XX.XX
Parameters (M):       XX.XX
FLOPs (G):            XX.XX
GPU Memory (MB):      XX.XX
==============================================================
```

## Architecture Details

### EfficientNet-B0 Specifications:
- **Input**: 224×224×3
- **Output feature map**: 7×7×1280
- **Depth**: 18 layers
- **Width multiplier**: 1.0
- **Resolution multiplier**: 1.0
- **Compound scaling**: α=1.2, β=1.1, γ=1.15

### Model Components:
1. **Backbone**: EfficientNet-B0 feature extractor
2. **Pattern Module**: SAMPPModule (Saliency-Aware Multi-Pattern Pooling)
3. **Attribute Branch**: Multi-task learning for composition attributes
4. **Score Prediction**: Distribution prediction for composition scores

## Compatibility

### Backward Compatibility:
The code maintains backward compatibility with ResNet backbones. To use ResNet:
```python
# In config.py, set:
backbone_type = 'resnet18'  # or 'resnet34', 'resnet50', 'resnet101'
```

### File Structure:
```
.
├── config.py              # Configuration (modified)
├── samp_net.py           # Network architecture (modified)
├── test.py               # Testing with benchmarks (modified)
├── train.py              # Training script (unchanged)
├── cadb_dataset.py       # Dataset loader (unchanged)
├── samp_module.py        # Pattern modules (unchanged)
├── requirements.txt      # Dependencies (updated)
└── README.md            # This file
```

## Benchmarking Notes

1. **FPS Measurement**: 
   - Warmup: 20 iterations
   - Measurement: 100 iterations
   - Batch size: 1 (for fair comparison)

2. **Memory Measurement**:
   - Reports peak memory allocation
   - Measured after inference on single batch

3. **FLOPs Calculation**:
   - Uses `thop` library
   - Includes both image and saliency inputs
   - Counts multiply-add operations

## Citation

If you use this code, please cite the original SAMP paper and acknowledge the EfficientNet architecture.

## License

Please refer to the original SAMP implementation for licensing information.
