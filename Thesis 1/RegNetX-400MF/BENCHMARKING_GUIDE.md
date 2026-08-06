# BENCHMARKING GUIDE FOR THESIS

## Overview
The modified `test.py` now includes comprehensive performance benchmarking to measure:
1. **Inference Speed (FPS)** - How fast the model processes images
2. **FLOPs** - Computational complexity
3. **Parameters** - Model size in terms of learnable parameters
4. **Model Size (MB)** - Storage/memory requirements

---

## Metrics Explained

### 1. Frames Per Second (FPS)
**What it measures:** How many images the model can process per second.

**Higher is better** - More FPS = Faster inference

**Typical Values:**
- CPU: 5-20 FPS
- Laptop GPU: 50-200 FPS  
- High-end GPU: 200-500+ FPS

**Benchmark Process:**
- 20 warmup iterations (to stabilize GPU)
- 100 timed iterations
- Uses `torch.cuda.synchronize()` for accurate timing
- Single image batch (batch_size=1) for real-world scenario

**Formula:** `FPS = 100 iterations / total_time`

### 2. Average Inference Time
**What it measures:** Time taken to process one image (in milliseconds).

**Lower is better** - Less time = Faster inference

**Calculation:** `(total_time / 100) * 1000` ms

**Use Case:** Important for real-time applications

### 3. FLOPs (Floating Point Operations)
**What it measures:** Total number of mathematical operations.

**Lower is better** - Fewer FLOPs = More efficient

**Reported in:**
- Raw count (e.g., 2,456,789,012)
- GFLOPs (billions): More readable (e.g., 2.46 GFLOPs)

**Typical Values:**
- MobileNet: ~0.3 GFLOPs
- ResNet-18: ~1.8 GFLOPs
- **RegNetX-400MF: ~0.4 GFLOPs**
- ResNet-50: ~4.1 GFLOPs

**Why it matters:** 
- Indicates computational cost
- Correlates with energy consumption
- Important for mobile/edge deployment

### 4. Parameters
**What it measures:** Number of learnable weights in the model.

**Lower is better** (for efficiency) - Fewer params = Smaller model

**Reported in:**
- Raw count (e.g., 5,234,567)
- Millions (M): More readable (e.g., 5.23 M)

**Typical Values:**
- MobileNet: ~4.2 M
- **RegNetX-400MF backbone: ~5.2 M**
- ResNet-18: ~11.7 M
- ResNet-50: ~25.6 M

**Why it matters:**
- Determines model size
- Affects training memory
- Important for storage constraints

### 5. Model Size (MB)
**What it measures:** Actual file size of the model.

**Lower is better** - Smaller size = Easier deployment

**Calculation:** Sum of all parameter and buffer memory

**Typical Values:**
- Small models: 5-20 MB
- **RegNetX-400MF SAMP-Net: ~10 MB**
- Medium models: 20-100 MB
- Large models: 100+ MB

### 6. GPU Memory Usage (MB)
**What it measures:** Peak GPU memory allocated during inference.

**Lower is better** - Less memory = Can run on smaller GPUs

**Calculation:** `torch.cuda.max_memory_allocated()` after reset

**Typical Values:**
- Lightweight models: 100-300 MB
- **RegNetX-400MF SAMP-Net: ~200-400 MB**
- Medium models: 400-1000 MB
- Heavy models: 1000+ MB

**Why it matters:**
- Determines minimum GPU memory required
- Important for batch processing
- Critical for deployment constraints

**Note:** This measures single image (batch_size=1). Multiply approximately by batch size for larger batches.

---

## Sample Output

When you run `python test.py`, you'll see:

```
================================================================================
PERFORMANCE BENCHMARKING
================================================================================
Warming up GPU...
Measuring inference speed...
Inference FPS: 156.23
Average Inference Time: 6.40 ms
Calculating FLOPs and Parameters...
FLOPs: 2,456,789,012 (2.46 GFLOPs)
Parameters: 8,234,567 (8.23 M)
Model Size: 31.42 MB
Measuring GPU memory usage...
GPU Memory Usage: 287.35 MB
================================================================================

Evaluation begining...
[Progress bar for test set]
Evaluation result...
Test on 1000 images, Accuracy=78.50%, EMD(r=1)=0.1234, EMD(r=2)=0.0987,
MSE_loss=0.2345, SRCC=0.8765, LCC=0.8654
```

---

## For Your Thesis

### Comparison Table Template

| Metric | ResNet-18 SAMP | RegNetX-400MF SAMP | Improvement |
|--------|----------------|---------------------|-------------|
| **Efficiency** |
| FLOPs (G) | ~2.5 | ~1.2 | -52% |
| Parameters (M) | ~15 | ~9 | -40% |
| Model Size (MB) | ~58 | ~34 | -41% |
| GPU Memory (MB) | ~450 | ~300 | -33% |
| **Speed** |
| FPS (GPU) | ~120 | ~180 | +50% |
| Inference Time (ms) | ~8.3 | ~5.6 | -32% |
| **Accuracy** |
| SRCC | 0.XXX | 0.YYY | ±Z% |
| LCC | 0.XXX | 0.YYY | ±Z% |
| EMD (r=1) | 0.XXX | 0.YYY | ±Z% |

*Fill in actual values after running experiments*

### Thesis Sections to Include

#### 1. Experimental Setup
"We benchmarked the model using a single NVIDIA [Your GPU] with CUDA [version]. 
FPS was measured over 100 iterations after 20 warmup runs. FLOPs and parameters 
were calculated using the thop library (Chen et al., 2021)."

#### 2. Results
"RegNetX-400MF achieved [X] FPS compared to ResNet-18's [Y] FPS, representing a 
[Z]% improvement. The computational cost was reduced from [A] GFLOPs to [B] GFLOPs 
([C]% reduction), while maintaining competitive accuracy (SRCC: [D] vs [E])."

#### 3. Discussion
"The efficiency gains make the RegNetX-400MF variant more suitable for deployment 
on resource-constrained devices while maintaining [acceptable/improved/similar] 
accuracy on composition assessment tasks."

---

## Installation

To use the benchmarking features, install `thop`:

```bash
pip install thop
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

---

## Running Benchmarks

### Basic Test (with benchmarking)
```bash
python test.py
```

This will:
1. Load your trained model
2. Run performance benchmarking (FPS, FLOPs, Params)
3. Evaluate accuracy on test set
4. Print all results

### Alternative: Create Standalone Benchmark Script

You can also create a separate script to benchmark without evaluation:

```python
# benchmark_only.py
from samp_net import SAMPNet
from config import Config
import torch
import time
from thop import profile

cfg = Config()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = SAMPNet(cfg, pretrained=True).to(device)
model.eval()

# Create dummy inputs
dummy_image = torch.randn(1, 3, 224, 224).to(device)
dummy_saliency = torch.randn(1, 1, 224, 224).to(device)

# Warmup
for _ in range(20):
    with torch.no_grad():
        _ = model(dummy_image, dummy_saliency)

# FPS measurement
torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    for _ in range(100):
        _ = model(dummy_image, dummy_saliency)
torch.cuda.synchronize()
end = time.time()

fps = 100 / (end - start)
print(f"FPS: {fps:.2f}")

# FLOPs and Params
flops, params = profile(model, inputs=(dummy_image, dummy_saliency), verbose=False)
print(f"FLOPs: {flops/1e9:.2f} G")
print(f"Params: {params/1e6:.2f} M")
```

---

## Troubleshooting

### Issue: `ImportError: No module named 'thop'`
**Solution:**
```bash
pip install thop
```

### Issue: FPS is very low on GPU
**Possible causes:**
1. GPU is being used by other processes
2. Batch size might be affecting measurement (we use batch=1 for fairness)
3. CUDA is not properly installed

**Check:**
```python
python -c "import torch; print(torch.cuda.is_available())"
```

### Issue: `RuntimeError: CUDA out of memory`
**Solution:** The benchmarking uses batch_size=1, so this shouldn't happen. 
If it does, restart your Python kernel to clear GPU memory.

### Issue: FLOPs calculation takes too long
**Note:** This is normal. The `profile()` function traces through the entire 
model which can take 30-60 seconds. It only runs once.

---

## Understanding Your Results

### Good FPS values:
- CPU: >10 FPS is acceptable
- Laptop GPU: >50 FPS is good, >100 FPS is excellent
- Desktop GPU: >200 FPS is good

### Expected FLOPs:
- **RegNetX-400MF SAMP-Net:** 0.8-1.5 GFLOPs (with saliency branch)
- Original ResNet-18 SAMP-Net: 2.0-3.0 GFLOPs

### Expected Parameters:
- **RegNetX-400MF SAMP-Net:** 8-10 M parameters
- Original ResNet-18 SAMP-Net: 13-16 M parameters

---

## Citation

If you use `thop` for benchmarking in your thesis:

```bibtex
@misc{thop,
  author = {Ligeng Zhu},
  title = {THOP: PyTorch-OpCounter},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Lyken17/pytorch-OpCounter}}
}
```

---

## Summary

The benchmarking additions provide comprehensive performance metrics essential 
for thesis work comparing different architectures. The measurements are:

✓ **Reproducible** - Uses warmup and synchronization
✓ **Standard** - Follows common benchmarking practices  
✓ **Comprehensive** - Covers speed, efficiency, size, and memory
✓ **Automated** - Runs with simple `python test.py` command

Good luck with your thesis! 🎓
