# QUICK REFERENCE: Benchmarking Metrics

## Installation
```bash
pip install thop
# OR
pip install -r requirements.txt
```

## Run Benchmarking
```bash
python test.py
```

## What You'll Get

### 1. FPS (Frames Per Second)
- **Measures:** Images processed per second
- **Higher = Better**
- **Expected:** 50-200 FPS on laptop GPU
- **Reported as:** `Inference FPS: 156.23`

### 2. Inference Time
- **Measures:** Time per image in milliseconds
- **Lower = Better**
- **Expected:** 5-20 ms on laptop GPU
- **Reported as:** `Average Inference Time: 6.40 ms`

### 3. FLOPs (Floating Point Operations)
- **Measures:** Computational complexity
- **Lower = Better** (more efficient)
- **Expected for RegNetX-400MF:** 0.8-1.5 GFLOPs
- **Expected for ResNet-18:** 2.0-3.0 GFLOPs
- **Reported as:** `FLOPs: 2,456,789,012 (2.46 GFLOPs)`

### 4. Parameters
- **Measures:** Learnable weights count
- **Lower = Better** (smaller model)
- **Expected for RegNetX-400MF:** 8-10 M
- **Expected for ResNet-18:** 13-16 M
- **Reported as:** `Parameters: 8,234,567 (8.23 M)`

### 5. Model Size
- **Measures:** Storage requirements in MB
- **Lower = Better**
- **Expected for RegNetX-400MF:** ~30-35 MB
- **Expected for ResNet-18:** ~55-60 MB
- **Reported as:** `Model Size: 31.42 MB`

### 6. GPU Memory Usage
- **Measures:** Peak GPU memory during inference
- **Lower = Better**
- **Expected for RegNetX-400MF:** ~200-400 MB
- **Expected for ResNet-18:** ~400-600 MB
- **Reported as:** `GPU Memory Usage: 287.35 MB`
- **Note:** Single image (batch_size=1)

## Sample Output
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
```

## For Your Thesis Table

| Metric | Value | Unit | Better |
|--------|-------|------|--------|
| FPS | ___ | images/sec | ↑ Higher |
| Inference Time | ___ | ms | ↓ Lower |
| FLOPs | ___ | GFLOPs | ↓ Lower |
| Parameters | ___ | M | ↓ Lower |
| Model Size | ___ | MB | ↓ Lower |
| GPU Memory | ___ | MB | ↓ Lower |
| SRCC | ___ | - | ↑ Higher |
| LCC | ___ | - | ↑ Higher |
| EMD (r=1) | ___ | - | ↓ Lower |
| Accuracy | ___ | % | ↑ Higher |

## Comparison Template

**Efficiency Improvements:**
- FLOPs reduced by: ___%
- Parameters reduced by: ___%
- Model size reduced by: ___%
- Inference speed increased by: ___%

**Accuracy Trade-off:**
- SRCC: ±___% 
- LCC: ±___%
- EMD: ±___%

## Common Issues

**No GPU detected:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**thop not installed:**
```bash
pip install thop
```

**Memory error:**
- Benchmarking uses batch_size=1 (minimal memory)
- Close other GPU applications
- Restart Python kernel

## Files Modified

✓ `test.py` - Added benchmarking code
✓ `requirements.txt` - Added `thop` dependency

## Files to Submit for Thesis

1. `test.py` (modified with benchmarking)
2. `samp_net.py` (RegNetX backbone)
3. `config.py` (laptop settings)
4. Results output / logs
5. Comparison tables (create from results)

---

**Remember:** Run on the same hardware for fair comparison!
