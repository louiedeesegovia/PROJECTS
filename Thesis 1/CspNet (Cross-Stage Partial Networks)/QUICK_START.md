# Quick Start Guide - CSPNet-SAMP Benchmark

## Overview
This is a modified version of SAMP-Net using CSPNet (Cross-Stage Partial Networks) as the backbone for image composition assessment benchmarking.

## Key Modifications
✅ **Backbone**: Changed from ResNet to CSPNet (CSPDarkNet53/CSPResNet50)
✅ **Batch Size**: 4 (optimized for benchmarking)
✅ **Workers**: 0 (for debugging and stability)
✅ **Epochs**: 50
✅ **Learning Rate**: 1e-4

## Files Included

### Core Files
- `cspnet_backbone.py` - CSPNet architecture implementation (NEW)
- `samp_net.py` - Modified network with CSPNet backbone
- `config.py` - Updated configuration with benchmark parameters
- `train.py` - Training script (unchanged)
- `test.py` - Testing script (unchanged)
- `cadb_dataset.py` - Dataset loader (unchanged)
- `samp_module.py` - SAMP modules (unchanged)

### Utility Files
- `compare_models.py` - Script to compare CSPNet variants
- `requirements.txt` - Dependencies
- `README.md` - Detailed documentation
- `QUICK_START.md` - This file

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Before You Start

1. **Verify Dataset Path**
   Edit `config.py` line 6:
   ```python
   dataset_path = '/workspace/composition/CADB_Dataset'
   ```
   Change to your actual dataset path.

2. **Choose CSPNet Variant**
   Edit `config.py` line 29:
   ```python
   cspnet_variant = 'cspdarknet53'  # or 'cspresnet50'
   ```
   
   **Recommendations**:
   - `cspdarknet53`: Lighter, faster, recommended for batch_size=4
   - `cspresnet50`: Deeper, more parameters, may need batch_size=2

## Usage

### 1. Compare Models (Optional)
```bash
python compare_models.py
```
This will show you:
- Parameter counts for each variant
- Memory usage
- Inference speed (if GPU available)
- Recommendations

### 2. Train Model
```bash
python train.py
```

**What happens**:
- Creates experiment directory: `./experiments/cspnet_cspdarknet53_samp_aaff_wemd/`
- Saves checkpoints every epoch
- Logs to TensorBoard
- Tests every epoch and saves best model

**Expected output**:
```
Experiment name cspnet_cspdarknet53_samp_aaff_wemd 

Create experiment directory: ./experiments/cspnet_cspdarknet53_samp_aaff_wemd
Training Epoch:0/50 Current Batch: 10/XXX EMD_Loss:0.XXXX Attribute_Loss:0.XXXX ACC:XX.XX% lr:0.000100
...
```

### 3. Test Model
```bash
python test.py
```

**Before testing**:
Update `test.py` line 68 with your checkpoint path:
```python
weight_file = './experiments/cspnet_cspdarknet53_samp_aaff_wemd/checkpoints/model-best.pth'
```

**Metrics reported**:
- Accuracy
- EMD (r=1 and r=2)
- MSE
- SRCC (Spearman Rank Correlation)
- LCC (Linear Correlation Coefficient)

## CSPNet Variants

### CSPDarkNet53 (Recommended for batch_size=4)
```python
cspnet_variant = 'cspdarknet53'
```
- **Output channels**: 1024
- **Parameters**: ~25M (full model)
- **Advantages**: Faster, less memory, good for benchmarking
- **Use when**: Starting experiments, limited GPU memory

### CSPResNet50
```python
cspnet_variant = 'cspresnet50'
```
- **Output channels**: 2048
- **Parameters**: ~45M (full model)
- **Advantages**: More capacity, potentially better performance
- **Use when**: Final models, more GPU memory available
- **Note**: May need `batch_size = 2`

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir=./experiments/cspnet_cspdarknet53_samp_aaff_wemd/logs
```

### Metrics Logged
- Train/EMD_Loss
- Train/Accuracy
- Train/AttributeLoss
- Train/lr
- Test/Average EMD (r=1, r=2)
- Test/Average MSE
- Test/Accuracy
- Test/SRCC
- Test/LCC

## Troubleshooting

### Out of Memory Error
```python
# In config.py
batch_size = 2  # Reduce from 4
# or
cspnet_variant = 'cspdarknet53'  # Use lighter variant
```

### Dataset Not Found
```python
# In config.py, update:
dataset_path = '/your/actual/path/to/CADB_Dataset'
```

### Slow Training
```python
# In config.py
num_workers = 4  # Increase if you have CPU cores available
```

## Expected Training Time

Approximate times per epoch (on common GPUs):
- **RTX 3090** (CSPDarkNet53): ~3-5 minutes
- **RTX 3090** (CSPResNet50): ~5-8 minutes
- **V100** (CSPDarkNet53): ~4-6 minutes
- **V100** (CSPResNet50): ~6-10 minutes

Total training (50 epochs):
- CSPDarkNet53: ~3-4 hours
- CSPResNet50: ~5-7 hours

## Results Location

After training, find your results in:
```
./experiments/cspnet_cspdarknet53_samp_aaff_wemd/
├── checkpoints/
│   ├── model-1.pth
│   ├── model-2.pth
│   ├── ...
│   └── model-best.pth  ← Best model based on EMD
├── logs/
│   └── events.out.tfevents...  ← TensorBoard logs
└── ..csv  ← Test metrics per epoch
```

## Benchmark Comparison

To compare with original ResNet baseline:
1. Train with CSPDarkNet53 (current setup)
2. Compare metrics in the generated CSV file
3. Use `compare_models.py` to get parameter counts

Expected differences:
- **Parameters**: CSPNet typically has fewer parameters
- **Speed**: CSPNet usually faster per epoch
- **Performance**: Should be comparable or better

## Tips for Best Results

1. **Learning Rate**: The default `1e-4` is good, but you can try:
   - `5e-5` for more stable training
   - `2e-4` for faster convergence (may be less stable)

2. **Batch Size**: 
   - Keep at 4 for fair benchmark comparison
   - Reduce to 2 if OOM errors

3. **Data Augmentation**: Already handled in dataset loader

4. **Checkpointing**: Best model saved automatically based on EMD

## Next Steps

1. ✅ Install dependencies
2. ✅ Verify dataset path
3. ✅ Run `compare_models.py` (optional)
4. ✅ Run `train.py`
5. ✅ Monitor with TensorBoard
6. ✅ Evaluate best model with `test.py`
7. ✅ Compare results with baseline

## Questions?

Check the detailed README.md for more information.

Good luck with your benchmark! 🚀
