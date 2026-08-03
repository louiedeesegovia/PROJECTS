# SAMP-Net with Inception-v1 Backbone - Benchmark Version

## Overview
This is a modified version of SAMP-Net (Saliency-Aware Multi-Pattern Pooling Network) for image composition assessment, adapted to use **Inception-v1 (GoogLeNet)** as the backbone instead of ResNet.

## Key Modifications

### 1. Backbone Change: ResNet → Inception-v1
- **File**: `samp_net.py`
- **Changes**:
  - Added `build_inception_v1()` function to build GoogLeNet backbone
  - Modified `SAMPNet` class to support both ResNet and Inception-v1
  - Inception-v1 outputs **1024 channels** (vs 512 for ResNet18/34, 2048 for ResNet50/101)
  - Backbone selection via `cfg.backbone_type = 'inception_v1'`

### 2. Configuration Updates
- **File**: `config.py`
- **Modified Parameters**:
  ```python
  batch_size = 4          # Changed from 16
  num_workers = 0         # Changed from 8
  max_epoch = 50          # Kept as is
  lr = 1e-4               # Kept as is
  backbone_type = 'inception_v1'  # NEW: backbone selection
  ```

### 3. Benchmarking Capabilities
- **File**: `test.py`
- **New Function**: `benchmark_model(model, device)`
- **Metrics Reported**:
  1. **Parameters**: Total, trainable, and non-trainable parameters
  2. **FLOPs**: Floating-point operations (requires `thop` library)
  3. **GPU Memory**: Peak memory allocated and reserved (MB)
  4. **Inference Speed**: FPS (frames per second) and average latency (ms)

### 4. Dependencies
- **File**: `requirements.txt`
- Added `thop` library for FLOPs calculation

## Architecture Details

### Inception-v1 Backbone
- **Architecture**: GoogLeNet (introduced in "Going Deeper with Convolutions", 2015)
- **Output Channels**: 1024 (from the last inception module)
- **Feature Map Size**: 7×7 (for 224×224 input)
- **Advantages**:
  - Efficient multi-scale feature extraction
  - Lower parameter count than ResNet50/101
  - Good balance between accuracy and speed

### Network Flow
```
Input Image (3×224×224)
    ↓
Inception-v1 Backbone
    ↓
Feature Maps (1024×7×7)
    ↓
SAMP Module (with Saliency)
    ↓
Attribute Branch + Composition Branch
    ↓
Output: Composition Score Distribution
```

## Usage

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Testing & Benchmarking
```bash
python test.py
```

**Output Example**:
```
============================================================
MODEL BENCHMARKING
============================================================

1. MODEL PARAMETERS:
   Total Parameters: 12,345,678
   Trainable Parameters: 12,345,678
   Non-trainable Parameters: 0

2. FLOPs (Floating Point Operations):
   FLOPs: 5.234G
   Params (from thop): 12.346M

3. GPU MEMORY USAGE:
   Peak Memory Allocated: 1234.56 MB
   Peak Memory Reserved: 1536.00 MB

4. INFERENCE SPEED:
   Iterations: 100
   Total Time: 2.3456 seconds
   Average Latency: 23.46 ms/image
   Throughput (FPS): 42.63 images/second

============================================================
```

### 3. Training
```bash
python train.py
```

**Training Configuration**:
- Batch size: 4
- Epochs: 50
- Learning rate: 1e-4
- Optimizer: Adam
- Workers: 0 (for compatibility)

### 4. Model Testing (with pretrained weights)
Uncomment these lines in `test.py`:
```python
weight_file = './pretrained_model/samp_net.pth'
model.load_state_dict(torch.load(weight_file))
evaluation_on_cadb(model, cfg)
```

## Benchmark Metrics Explained

### 1. Parameters
- **Total Parameters**: All learnable weights in the model
- **Trainable Parameters**: Parameters updated during training
- **Typical Range**: 5M - 50M for modern CNNs

### 2. FLOPs (Floating-Point Operations)
- Measures computational complexity
- **Lower is better** for efficiency
- **Typical Range**: 1G - 20G FLOPs for image classification

### 3. GPU Memory
- **Peak Allocated**: Actual memory used by tensors
- **Peak Reserved**: Memory reserved by CUDA allocator
- **Importance**: Determines maximum batch size

### 4. Inference Speed (FPS)
- **FPS**: Images processed per second
- **Latency**: Time to process one image
- **Higher FPS = Lower Latency**
- Important for real-time applications

## Comparison: Inception-v1 vs ResNet

| Metric | Inception-v1 | ResNet-18 | ResNet-50 |
|--------|--------------|-----------|-----------|
| Parameters | ~6M | ~11M | ~25M |
| Output Channels | 1024 | 512 | 2048 |
| FLOPs (approx) | ~1.5G | ~1.8G | ~4.1G |
| Speed | Fast | Fast | Medium |
| Accuracy | Good | Good | Better |

## File Structure
```
.
├── config.py              # Configuration (modified)
├── samp_net.py           # Network architecture (modified)
├── test.py               # Testing & benchmarking (modified)
├── train.py              # Training script (original)
├── cadb_dataset.py       # Dataset loader (original)
├── samp_module.py        # Pattern pooling modules (original)
├── requirements.txt      # Dependencies (updated)
└── README.md            # This file
```

## Notes

### Why Inception-v1?
1. **Efficiency**: Good accuracy with fewer parameters
2. **Multi-scale**: Inception modules capture features at different scales
3. **Proven**: Well-established architecture with pretrained weights
4. **Benchmark**: Popular choice for comparing model performance

### Training Tips
- Start with pretrained ImageNet weights (`pretrained=True`)
- Monitor GPU memory usage (reduce batch size if OOM)
- Use learning rate scheduling (already implemented)
- Check tensorboard logs in `experiments/*/logs/`

### Benchmarking Notes
- FPS measured on single image (batch size = 1)
- Warmup iterations ensure stable GPU clock speeds
- Memory usage may vary with batch size
- FLOPs are input-size dependent (measured at 224×224)

## Citation
If you use this code, please cite the original SAMP-Net paper:
```
@article{samp_net,
  title={SAMP: Composition Assessment with Multi-Pattern Pooling},
  author={...},
  journal={...},
  year={...}
}
```

## License
Same as the original SAMP-Net implementation.

## Troubleshooting

### Issue: Out of Memory (OOM)
**Solution**: Reduce `batch_size` in `config.py`

### Issue: thop import error
**Solution**: `pip install thop`

### Issue: Slow training
**Solution**: Increase `num_workers` in `config.py` (if your system supports it)

### Issue: CUDA out of memory during benchmark
**Solution**: Benchmarking uses batch_size=1, but if still OOM, your GPU may be too small

## Contact
For questions or issues, please refer to the original SAMP-Net repository.
