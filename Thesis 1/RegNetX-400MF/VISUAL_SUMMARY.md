# BENCHMARKING METRICS SUMMARY - VISUAL GUIDE

## 📊 Complete Metrics Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                  PERFORMANCE BENCHMARKING                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  🚀 SPEED METRICS                                              │
│  ├─ Inference FPS: ___.___ images/sec     [Higher ✓]          │
│  └─ Avg Inference Time: ___.___ ms         [Lower ✓]          │
│                                                                 │
│  💻 EFFICIENCY METRICS                                         │
│  ├─ FLOPs: ___,___,___ (_.__ GFLOPs)      [Lower ✓]          │
│  ├─ Parameters: ___,___ (_.__ M)           [Lower ✓]          │
│  ├─ Model Size: ___.___ MB                 [Lower ✓]          │
│  └─ GPU Memory: ___.___ MB                 [Lower ✓]          │
│                                                                 │
│  🎯 ACCURACY METRICS                                           │
│  ├─ Accuracy: __.__%                       [Higher ✓]         │
│  ├─ SRCC: 0.____                           [Higher ✓]         │
│  ├─ LCC: 0.____                            [Higher ✓]         │
│  ├─ EMD (r=1): 0.____                      [Lower ✓]          │
│  └─ EMD (r=2): 0.____                      [Lower ✓]          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🔍 What Each Metric Means

### 1. FPS (Frames Per Second)
```
┌──────────────────────────┐
│ How many images can be   │
│ processed in 1 second    │
├──────────────────────────┤
│ Example: 150 FPS         │
│ = 150 images/second      │
│ = 0.0067 seconds/image   │
└──────────────────────────┘
```

### 2. Average Inference Time
```
┌──────────────────────────┐
│ Time to process 1 image  │
│ in milliseconds          │
├──────────────────────────┤
│ Example: 6.67 ms         │
│ = 0.00667 seconds        │
│ = 1/150 second           │
└──────────────────────────┘
```

### 3. FLOPs (Floating Point Operations)
```
┌──────────────────────────┐
│ Total math operations    │
│ needed for 1 inference   │
├──────────────────────────┤
│ Example: 1.2 GFLOPs      │
│ = 1,200,000,000 ops      │
│                          │
│ Lower = More Efficient   │
└──────────────────────────┘
```

### 4. Parameters
```
┌──────────────────────────┐
│ Number of learnable      │
│ weights in the model     │
├──────────────────────────┤
│ Example: 8.77 M          │
│ = 8,770,000 weights      │
│                          │
│ Lower = Smaller Model    │
└──────────────────────────┘
```

### 5. Model Size
```
┌──────────────────────────┐
│ Disk space needed to     │
│ store the model file     │
├──────────────────────────┤
│ Example: 33.45 MB        │
│ ≈ 8.77M params × 4 bytes │
│                          │
│ Lower = Less Storage     │
└──────────────────────────┘
```

### 6. GPU Memory Usage
```
┌──────────────────────────┐
│ RAM used on GPU during   │
│ inference (batch_size=1) │
├──────────────────────────┤
│ Example: 287 MB          │
│                          │
│ Includes:                │
│ • Model weights          │
│ • Activations            │
│ • Intermediate tensors   │
│                          │
│ Lower = Less GPU RAM     │
└──────────────────────────┘
```

## 📈 Expected Ranges

```
Metric                  RegNetX-400MF         ResNet-18
─────────────────────────────────────────────────────────
FPS (laptop GPU)        100-200               80-150
Inference Time (ms)     5-10                  7-13
FLOPs (G)              0.8-1.5               2.0-3.0
Parameters (M)          8-10                  13-16
Model Size (MB)         30-40                 50-65
GPU Memory (MB)         200-400               400-600
```

## 🎓 For Your Thesis

### Comparison Table Format

```
┌──────────────────┬──────────────┬──────────────┬────────────┐
│ Metric           │ ResNet-18    │ RegNetX-400MF│ Change (%) │
├──────────────────┼──────────────┼──────────────┼────────────┤
│ FPS              │    120       │     180      │    +50%    │
│ Inference (ms)   │    8.3       │     5.6      │    -33%    │
│ FLOPs (G)        │    2.5       │     1.2      │    -52%    │
│ Parameters (M)   │    15        │      9       │    -40%    │
│ Model Size (MB)  │    58        │     34       │    -41%    │
│ GPU Memory (MB)  │   450        │    300       │    -33%    │
├──────────────────┼──────────────┼──────────────┼────────────┤
│ SRCC             │   0.XXX      │   0.YYY      │   ±Z%      │
│ LCC              │   0.XXX      │   0.YYY      │   ±Z%      │
│ EMD (r=1)        │   0.XXX      │   0.YYY      │   ±Z%      │
└──────────────────┴──────────────┴──────────────┴────────────┘
```

## 🔧 How It Works in test.py

```python
# 1. Warmup (20 iterations)
for _ in range(20):
    _ = model(dummy_image, dummy_saliency)

# 2. FPS Measurement (100 iterations)
start = time.time()
for _ in range(100):
    _ = model(dummy_image, dummy_saliency)
end = time.time()
fps = 100 / (end - start)

# 3. FLOPs and Parameters
flops, params = profile(model, inputs=(dummy_image, dummy_saliency))

# 4. Model Size
size_mb = (param_size + buffer_size) / 1024**2

# 5. GPU Memory
torch.cuda.reset_peak_memory_stats()
_ = model(dummy_image, dummy_saliency)
peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
```

## 📝 Sample Complete Output

```
================================================================================
PERFORMANCE BENCHMARKING
================================================================================
Warming up GPU...
Measuring inference speed...
Inference FPS: 156.23
Average Inference Time: 6.40 ms
Calculating FLOPs and Parameters...
FLOPs: 1,234,567,890 (1.23 GFLOPs)
Parameters: 8,765,432 (8.77 M)
Model Size: 33.45 MB
Measuring GPU memory usage...
GPU Memory Usage: 287.35 MB
================================================================================

Evaluation begining...
100%|████████████████████████████████| 250/250 [00:15<00:00, 16.23it/s]
Evaluation result...
Test on 1000 images, Accuracy=78.50%, EMD(r=1)=0.1234, EMD(r=2)=0.0987,
MSE_loss=0.2345, SRCC=0.8765, LCC=0.8654
```

## ✅ Checklist for Running

- [ ] Install thop: `pip install thop`
- [ ] Upgrade PyTorch: Use torchvision >= 0.13.0
- [ ] Have GPU available (or will run on CPU, slower)
- [ ] Load trained model weights
- [ ] Run: `python test.py`
- [ ] Record all metrics for thesis
- [ ] Compare with baseline (ResNet-18)

## 🎯 Key Takeaways for Thesis

### Efficiency Wins
✓ **Lower FLOPs** = Less computation needed
✓ **Fewer Parameters** = Smaller model
✓ **Less GPU Memory** = Can run on cheaper hardware
✓ **Higher FPS** = Faster real-time processing

### The Trade-off
- Efficiency ⬆️ (RegNetX is more efficient)
- Accuracy ≈ (Should be similar, document actual results)

### Why This Matters
Your thesis shows that modern efficient architectures (RegNetX) can achieve 
comparable performance to older standard architectures (ResNet) while being:
- 40-50% more efficient
- 30-50% faster
- Suitable for resource-constrained deployment

## 📊 Visualization Ideas for Thesis

### Bar Chart: Efficiency Comparison
```
FLOPs (Lower is Better)
ResNet-18   ████████████████████████████ 2.5G
RegNetX-400 ████████████ 1.2G
            └─────┴─────┴─────┴─────┘
            0    1     2     3     4

Parameters (Lower is Better)
ResNet-18   ████████████████████████████████ 15M
RegNetX-400 ████████████████████ 9M
            └─────┴─────┴─────┴─────┘
            0    5     10    15    20
```

### Line Chart: Speed Comparison
```
FPS (Higher is Better)
200 │                    •
    │               •
150 │          •
    │     •
100 │•
    └─────────────────────────
     ResNet-18  RegNetX-400
```

## 🚀 Quick Commands

```bash
# Install dependencies
pip install thop
pip install torch==1.12.1 torchvision==0.13.1

# Run benchmarking
python test.py

# Just benchmarking (no full test)
python -c "from test import *; cfg = Config(); model = SAMPNet(cfg).cuda(); evaluation_on_cadb(model, cfg)" 2>&1 | head -20
```

---

**Remember:** All metrics are measured on single image (batch_size=1) 
for fair comparison and real-world inference scenarios.

Good luck with your thesis! 📚🎓
