# SAMP-Net with RegNetX-400MF Backbone

## Overview
This is a modified version of the SAMP-Net (Saliency-Aware Multi-Pattern Pooling Network) for composition assessment, adapted to use **RegNetX-400MF** as the backbone instead of ResNet. This modification is optimized for laptop training with reduced computational requirements.

## Key Modifications

### 1. Backbone Change: ResNet → RegNetX-400MF
**Original:** ResNet-18 (512 channels output)
**Modified:** RegNetX-400MF (400 channels output)

**Benefits of RegNetX-400MF:**
- **Fewer parameters:** ~5.2M parameters vs ResNet-18's ~11.7M
- **Faster training:** More efficient architecture designed for mobile/edge devices
- **Lower memory footprint:** Better suited for laptop GPUs with limited VRAM
- **Competitive performance:** State-of-the-art efficiency-accuracy trade-off

### 2. Training Settings (Optimized for Laptop)
```python
batch_size = 4        # Reduced from 16 for memory efficiency
num_workers = 0       # Set to 0 to avoid multiprocessing issues on laptops
max_epoch = 50        # Maintained as requested
lr = 1e-4            # Maintained as requested
```

### 3. Model Architecture Changes

**File: `samp_net.py`**
- Changed `build_resnet()` to `build_regnet()`
- Updated `input_channel` from 512 (ResNet-18) to 400 (RegNetX-400MF)
- Backbone now uses: `torchvision.models.regnet_x_400mf()`

**File: `config.py`**
- Replaced `resnet_layers = 18` with `backbone = 'regnetx_400mf'`
- Updated experiment prefix from 'resnet18' to 'regnetx400mf'
- Adjusted batch_size and num_workers for laptop compatibility

## File Structure
```
.
├── samp_net.py          # Modified network architecture with RegNetX
├── config.py            # Modified configuration for laptop training
├── train.py             # Training script (unchanged)
├── test.py              # Testing script (unchanged)
├── cadb_dataset.py      # Dataset loader (unchanged)
├── samp_module.py       # Pattern modules (unchanged)
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

**Note:** The requirements.txt uses PyTorch 1.9.1. For newer systems, you may need to update:
```bash
# For CUDA 11.x systems
pip install torch==1.12.1 torchvision==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu113

# For CPU-only (if no GPU)
pip install torch==1.12.1 torchvision==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
```

### 2. Dataset Setup
Ensure your CADB dataset is at the path specified in `config.py`:
```python
dataset_path = '/workspace/composition/CADB_Dataset'
```

Dataset should contain:
- `images/` - Image files
- `composition_scores.json` - Score annotations
- `split.json` - Train/test split
- `composition_attributes.json` - Attribute annotations
- `emdloss_weight.json` - Sample weights for weighted EMD loss

## Usage

### Training
```bash
python train.py
```

**Training Features:**
- Automatic experiment directory creation
- TensorBoard logging
- Model checkpoints saved every epoch
- Best model saved based on validation EMD score
- CSV metrics logging

### Testing
```bash
python test.py
```

**Evaluation Metrics:**
- Accuracy (threshold-based classification)
- EMD Loss (r=1 and r=2)
- MSE Loss
- SRCC (Spearman Rank Correlation)
- LCC (Linear Correlation Coefficient)

### Monitor Training
```bash
tensorboard --logdir experiments/regnetx400mf_samp_aaff_wemd/logs
```

## Model Architecture Details

### RegNetX-400MF Backbone
```
Input (3, 224, 224)
    ↓
Stem (Conv + BN + ReLU)
    ↓
Stage 1 (32 channels)
    ↓
Stage 2 (64 channels)
    ↓
Stage 3 (160 channels)
    ↓
Stage 4 (400 channels) → Feature Map (400, 7, 7)
    ↓
SAMP Modules (Pattern-based pooling)
    ↓
Score Prediction (5 classes)
```

### SAMP Components
1. **Multi-Pattern Pooling (MPP):** 8 different composition patterns
2. **Saliency-Aware:** Integrates saliency maps for attention
3. **Attribute Branch:** Multi-task learning with composition attributes
4. **Channel Attention:** Adaptive feature fusion

## Memory Requirements

### Estimated GPU Memory (per configuration)
- **Batch Size 4:** ~3-4 GB VRAM
- **Batch Size 8:** ~6-7 GB VRAM (if you have more memory)
- **Batch Size 16:** ~12-14 GB VRAM (original setting)

### Tips for Limited Memory
1. Reduce `batch_size` to 2 if needed
2. Use mixed precision training (add to train.py):
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   ```
3. Gradient accumulation for effective larger batch size
4. Close other applications to free up RAM

## Expected Performance

### Training Time (Laptop estimates)
- **Per Epoch:** ~15-30 minutes (depending on laptop specs)
- **Total Training (50 epochs):** ~12-25 hours
- **With GPU:** Significantly faster than CPU-only

### Model Size
- **RegNetX-400MF SAMP-Net:** ~8-10 MB
- **Original ResNet-18 SAMP-Net:** ~15-18 MB

### Benchmark Comparison
| Model | Parameters | FLOPs | Memory | Training Speed |
|-------|-----------|-------|---------|----------------|
| ResNet-18 | ~11.7M | 1.8G | High | Baseline |
| RegNetX-400MF | ~5.2M | 0.4G | Low | ~1.5x faster |

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# In config.py, reduce:
batch_size = 2
```

**2. Import Errors**
```bash
# Ensure all files are in the same directory
# Check Python path
export PYTHONPATH="${PYTHONPATH}:."
```

**3. Dataset Path Error**
```python
# Update in config.py:
dataset_path = 'YOUR_ACTUAL_PATH/CADB_Dataset'
```

**4. Slow Training on CPU**
```python
# In train.py, you can reduce model complexity:
# Modify config.py:
pattern_list = [1, 2, 3, 4]  # Use fewer patterns
```

## Experiment Tracking

Results are saved in:
```
experiments/
└── regnetx400mf_samp_aaff_wemd/
    ├── checkpoints/
    │   ├── model-1.pth
    │   ├── model-2.pth
    │   └── model-best.pth
    ├── logs/
    │   └── tensorboard_events
    └── ../regnetx400mf_samp_aaff_wemd.csv
```

## Citation

If you use this modified code for your thesis, please cite:
1. Original SAMP paper (for the method)
2. RegNet paper (for the backbone)
3. CADB dataset paper

```bibtex
@article{regnet2020,
  title={Designing Network Design Spaces},
  author={Radosavovic, Ilija and Kosaraju, Raj Prateek and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
  journal={CVPR},
  year={2020}
}
```

## License
Same as original SAMP-Net code.

## Acknowledgments
- Original SAMP-Net implementation
- PyTorch and torchvision for RegNet implementation
- CADB dataset creators

## Contact
For thesis-related questions, refer to your advisor.
For technical issues with this modification, check PyTorch documentation.
