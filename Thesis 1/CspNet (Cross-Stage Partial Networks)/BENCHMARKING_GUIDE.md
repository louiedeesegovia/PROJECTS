# Benchmarking Guide - CSPNet-SAMP

This document explains how to use the comprehensive benchmarking features added to the CSPNet-SAMP codebase.

## Overview

The codebase now includes detailed benchmarking capabilities to measure:
- **Model Parameters** - Total and trainable parameters
- **FLOPs** - Computational complexity (floating point operations)
- **Inference Speed** - FPS and latency measurements
- **GPU Memory Usage** - Memory consumption during inference
- **Training Time** - Per-epoch and total training time

## Required Dependencies

Install the additional dependency for FLOPs calculation:
```bash
pip install thop
```

Or use the updated requirements.txt:
```bash
pip install -r requirements.txt
```

## Benchmarking Features

### 1. Testing Script (`test.py`)

The `test.py` script now automatically benchmarks the model before evaluation.

**What it measures:**
- Model parameters (total and trainable)
- FLOPs (using thop library)
- Inference speed (FPS) for single images
- Batch processing throughput
- GPU memory usage (single and batch)

**Usage:**
```bash
python test.py
```

**Output Example:**
```
============================================================
MODEL BENCHMARKING
============================================================

1. Model Parameters:
   Total Parameters: 25,123,456
   Trainable Parameters: 25,123,456
   Model Size: 95.84 MB

2. Computational Complexity:
   FLOPs: 8,234,567,890 (8.23 GFLOPs)
   Params (from thop): 25,123,456

3. Inference Speed:
   Inference Time: 12.45 ms per image
   Inference FPS: 80.32 images/sec
   Throughput (batch=4): 321.28 images/sec

4. GPU Memory Usage:
   Peak Memory Allocated: 456.78 MB
   Memory Reserved: 512.00 MB

5. Batch Processing (batch_size=4):
   Batch Time: 45.67 ms per batch
   Batch Throughput: 87.54 images/sec
   Batch Memory: 789.12 MB
============================================================
```

### 2. Training Script (`train.py`)

The `train.py` script now tracks and reports training time statistics.

**What it measures:**
- Per-epoch training time
- Total training time
- Average epoch time
- Estimated remaining time
- Training time logged to TensorBoard

**Features:**
- Real-time training time estimates
- Epoch time tracking
- Summary statistics at end of training
- Training time saved to CSV results

**Usage:**
```bash
python train.py
```

**Output Example:**
```
Training started at: 2024-02-03 10:30:00
Total epochs: 50
============================================================

Training Epoch:0/50 Current Batch: 10/200 EMD_Loss:0.1234 ...

Epoch 1 completed in 3.45 minutes
Average epoch time: 3.45 minutes
Total training time: 0.06 hours
Estimated remaining time: 2.82 hours

...

============================================================
TRAINING COMPLETED
============================================================
Total training time: 2.88 hours
Average epoch time: 3.46 minutes
Fastest epoch: 3.21 minutes
Slowest epoch: 3.78 minutes
Training finished at: 2024-02-03 13:18:00
============================================================
```

**CSV Output:**
The results CSV now includes an "Epoch Time (min)" column and a training summary section at the bottom.

### 3. Benchmark Summary Script (`benchmark_summary.py`)

A comprehensive standalone benchmarking script that generates a complete report.

**Usage:**
```bash
# Benchmark architecture only
python benchmark_summary.py

# Benchmark with trained weights (edit script to specify checkpoint path)
python benchmark_summary.py
```

**Features:**
- Complete model configuration summary
- Detailed parameter counts
- FLOPs calculation
- Inference speed tests (single and batch)
- Memory usage analysis
- Results saved to JSON file

**Output:**
```
================================================================================
COMPREHENSIVE BENCHMARK REPORT - CSPNet-SAMP
================================================================================

--------------------------------------------------------------------------------
1. MODEL CONFIGURATION
--------------------------------------------------------------------------------
Backbone: cspdarknet53
Image Size: 224
Batch Size: 4
Learning Rate: 0.0001
Max Epochs: 50

--------------------------------------------------------------------------------
2. MODEL SIZE
--------------------------------------------------------------------------------
Total Parameters: 25,123,456
Trainable Parameters: 25,123,456
Model Size: 95.84 MB

... (continues with all metrics)

================================================================================
SUMMARY TABLE
================================================================================
Metric                                                              Value
--------------------------------------------------------------------------------
Backbone                                                   cspdarknet53
Total Parameters                                             25,123,456
Model Size (MB)                                                   95.84
FLOPs (GFLOPs)                                                     8.23
Inference Time - Single (ms)                                      12.45
FPS - Single                                                      80.32
Inference Time - Batch (ms)                                       45.67
Throughput - Batch (img/s)                                        87.54
GPU Memory - Single (MB)                                         456.78
GPU Memory - Batch (MB)                                          789.12
================================================================================

Benchmark results saved to: ./experiments/.../benchmark_results.json
```

## Metrics Explained

### Parameters
- **Total Parameters**: All model parameters
- **Trainable Parameters**: Parameters that will be updated during training
- **Model Size**: Memory size in MB (params × 4 bytes)

### FLOPs (Floating Point Operations)
- Measures computational complexity
- **GFLOPs**: Billion floating point operations
- Lower is faster, but may sacrifice accuracy

### Inference Speed
- **FPS**: Frames (images) per second
- **Inference Time**: Time to process one image/batch (milliseconds)
- **Throughput**: Total images processed per second

### Memory Usage
- **Allocated**: Actual GPU memory used
- **Reserved**: Memory reserved by PyTorch
- Measured during forward pass only

### Training Time
- **Epoch Time**: Time to complete one full pass through training data
- **Total Time**: Cumulative training time
- **Average/Min/Max**: Statistics across all epochs

## TensorBoard Visualization

Training time metrics are logged to TensorBoard:
```bash
tensorboard --logdir=./experiments/cspnet_cspdarknet53_samp_aaff_wemd/logs
```

**Available metrics:**
- `Train/EpochTime` - Time per epoch (minutes)
- `Train/TotalTime` - Cumulative time (hours)
- Plus all existing training/testing metrics

## CSV Results Format

The CSV file now includes:
```csv
epoch,Accuracy,EMD r=1,EMD r=2,MSE,SRCC,LCC,Epoch Time (min)
0,0.7234,0.1234,0.0987,0.0456,0.8123,0.8234,3.45
1,0.7456,0.1123,0.0876,0.0423,0.8234,0.8345,3.52
...

Training Summary,,,,,,
Total Training Time (hours),,,,,,,2.88
Average Epoch Time (min),,,,,,,3.46
Fastest Epoch (min),,,,,,,3.21
Slowest Epoch (min),,,,,,,3.78
```

## Benchmark Comparison Workflow

To compare different configurations:

1. **Train Model A** (e.g., CSPDarkNet53):
   ```bash
   # Set cspnet_variant = 'cspdarknet53' in config.py
   python train.py
   python test.py  # Generates benchmark report
   ```

2. **Train Model B** (e.g., CSPResNet50):
   ```bash
   # Set cspnet_variant = 'cspresnet50' in config.py
   python train.py
   python test.py  # Generates benchmark report
   ```

3. **Compare Results**:
   - Check CSV files in `./experiments/` directory
   - Compare benchmark_results.json files
   - Review TensorBoard logs

## Example Benchmark Results

### CSPDarkNet53 (Expected)
- Parameters: ~25M
- FLOPs: ~8 GFLOPs
- FPS: ~80-100 (RTX 3090)
- Memory: ~500 MB (single image)
- Epoch Time: ~3-5 minutes

### CSPResNet50 (Expected)
- Parameters: ~45M
- FLOPs: ~15 GFLOPs
- FPS: ~50-70 (RTX 3090)
- Memory: ~800 MB (single image)
- Epoch Time: ~5-8 minutes

*Note: Actual numbers will vary based on GPU, CUDA version, and system configuration.*

## Tips for Accurate Benchmarking

1. **Warm Up GPU**: The scripts include warmup iterations - don't skip them
2. **Close Other Programs**: Minimize GPU usage from other applications
3. **Consistent Settings**: Use same batch_size, image_size for comparisons
4. **Multiple Runs**: Run benchmarks multiple times and average results
5. **Check Temperature**: Ensure GPU isn't thermal throttling

## Troubleshooting

### thop not working
```bash
pip install thop --upgrade
```
If still fails, FLOPs will be skipped but other metrics will work.

### CUDA out of memory during benchmarking
Reduce batch_size in config.py temporarily for benchmarking.

### Inconsistent timing measurements
- Close TensorBoard and other GPU applications
- Increase warmup iterations in benchmark functions
- Check GPU is not being throttled

## Files Modified

- `test.py` - Added comprehensive benchmarking before evaluation
- `train.py` - Added training time tracking and reporting
- `requirements.txt` - Added thop dependency
- `benchmark_summary.py` - New standalone benchmarking script

## Integration with Research Papers

When reporting results in papers, use these metrics:

**Table 1: Model Complexity**
| Model | Params (M) | FLOPs (G) | Size (MB) |
|-------|-----------|-----------|-----------|
| CSPDarkNet53-SAMP | X.X | X.X | X.X |

**Table 2: Inference Speed**
| Model | FPS | Time (ms) | Memory (MB) |
|-------|-----|-----------|-------------|
| CSPDarkNet53-SAMP | X.X | X.X | X.X |

**Table 3: Training Efficiency**
| Model | Epoch Time (min) | Total Time (h) |
|-------|-----------------|----------------|
| CSPDarkNet53-SAMP | X.X | X.X |

All metrics are automatically collected and can be extracted from:
- Console output
- benchmark_results.json
- Training CSV file
- TensorBoard logs
