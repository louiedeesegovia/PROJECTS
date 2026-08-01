# Quick Start Guide - EfficientNet-B0 SAMP

## Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

Note: For PyTorch 1.9.1 with CUDA 11.1, you may need:
```bash
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

For newer PyTorch versions, update `requirements.txt` accordingly.

## Quick Verification

**Verify the model can be created and runs correctly**:
```bash
python verify_model.py
```

Expected output:
```
============================================================
Testing EfficientNet-B0 SAMP Model Creation
============================================================

Configuration:
  Backbone: efficientnet_b0
  Batch size: 4
  Image size: 224
  Learning rate: 0.0001
  Max epochs: 50
  Num workers: 0

Creating model...

Model Statistics:
  Total parameters: X,XXX,XXX
  Trainable parameters: X,XXX,XXX
  Parameters (M): XX.XX

Testing forward pass...

Output shapes:
  Pattern weights: torch.Size([2, 8])
  Attributes: torch.Size([2, 6])
  Scores: torch.Size([2, 5])

============================================================
✓ Model creation and forward pass successful!
============================================================
```

## Compare Backbones

**Compare EfficientNet-B0 vs ResNet-18**:
```bash
python compare_backbones.py
```

This will benchmark both architectures and show:
- Parameters (millions)
- FLOPs (billions)
- Inference FPS
- GPU Memory usage

## Training

**Start training**:
```bash
python train.py
```

Training parameters (from config.py):
- Batch size: 4
- Learning rate: 1e-4
- Max epochs: 50
- Optimizer: Adam
- Weight decay: 5e-5

The script will:
1. Create experiment directory
2. Save checkpoints every epoch
3. Log metrics to TensorBoard
4. Save best model based on EMD loss

## Testing and Benchmarking

**Run full evaluation with benchmarks**:
```bash
python test.py
```

This will:
1. Load the model
2. Run benchmark tests (FPS, FLOPs, Parameters, Memory)
3. Evaluate on test dataset
4. Print comprehensive results

### Expected Benchmark Output:

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
```

## Key Configuration Parameters

Edit `config.py` to modify:

```python
# Dataset
dataset_path = '/workspace/composition/CADB_Dataset'

# Training
batch_size = 4
max_epoch = 50
lr = 1e-4
num_workers = 0

# Model
backbone_type = 'efficientnet_b0'  # or 'resnet18', 'resnet34', etc.
image_size = 224
dropout = 0.5
pool_dropout = 0.5

# Features
use_weighted_loss = True
use_attribute = True
use_channel_attention = True
use_saliency = True
use_multipattern = True
use_pattern_weight = True

# Patterns (composition patterns to use)
pattern_list = [1, 2, 3, 4, 5, 6, 7, 8]
```

## Switching Between Backbones

To switch from EfficientNet-B0 to ResNet:

1. Edit `config.py`:
```python
backbone_type = 'resnet18'  # or 'resnet34', 'resnet50', 'resnet101'
```

2. The model will automatically adjust the architecture

## File Structure

```
.
├── config.py              # Configuration file
├── samp_net.py           # Model architecture with EfficientNet-B0
├── samp_module.py        # Pattern pooling modules
├── cadb_dataset.py       # Dataset loader
├── train.py              # Training script
├── test.py               # Testing with benchmarks
├── verify_model.py       # Quick verification script
├── compare_backbones.py  # Backbone comparison script
├── requirements.txt      # Python dependencies
├── README.md            # Full documentation
└── QUICKSTART.md        # This file
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in `config.py` (try 2 or 1)
- Reduce `image_size` (try 192 instead of 224)

### Module Not Found
- Ensure all files are in the same directory
- Check that `requirements.txt` dependencies are installed

### Dataset Not Found
- Update `dataset_path` in `config.py` to point to your CADB dataset
- Ensure the dataset has the following structure:
  ```
  CADB_Dataset/
  ├── images/
  ├── composition_scores.json
  ├── split.json
  ├── composition_attributes.json
  └── emdloss_weight.json
  ```

### thop Installation Issues
- If `thop` fails to install, you can comment out FLOPs calculation
- The rest of the benchmarking will still work

## Monitoring Training

Use TensorBoard to monitor training:
```bash
tensorboard --logdir=./experiments/
```

Then open http://localhost:6006 in your browser.

## Additional Resources

- Original SAMP paper: [Add citation]
- EfficientNet paper: https://arxiv.org/abs/1905.11946
- CADB Dataset: [Add link]

## Support

For issues or questions:
1. Check the README.md for detailed documentation
2. Verify all dependencies are correctly installed
3. Ensure dataset path is correctly configured
