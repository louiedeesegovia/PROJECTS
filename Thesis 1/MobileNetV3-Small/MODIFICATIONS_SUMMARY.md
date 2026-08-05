# Code Modifications Summary - MobileNetV3-Small Backbone

## Overview
This document summarizes all changes made to convert SAMP-Net from ResNet backbone to MobileNetV3-Small, along with the requested configuration changes and benchmarking additions.

---

## Modified Files

### 1. `config.py`
**Changes:**
- `batch_size = 4` (was 16)
- `num_workers = 0` (was 8)
- `max_epoch = 50` (unchanged, as requested)
- `lr = 1e-4` (unchanged, as requested)
- Added `backbone_type = 'mobilenetv3_small'` parameter
- Updated experiment prefix from `'resnet{}'.format(resnet_layers)` to `'mobilenetv3_small'`

**Reason:** Configure for MobileNetV3-Small training with requested parameters.

---

### 2. `samp_net.py`
**Changes:**

#### Added Function:
```python
def build_mobilenetv3_small(pretrained=True):
    """Build MobileNetV3-Small backbone"""
    mobilenet = models.mobilenet_v3_small(pretrained=pretrained)
    backbone = mobilenet.features
    return backbone
```

#### Modified `SAMPNet.__init__()`:
```python
# Set input_channel based on backbone type
if backbone_type == 'mobilenetv3_small':
    input_channel = 576  # MobileNetV3-Small output channels
    self.backbone = build_mobilenetv3_small(pretrained=pretrained)
else:
    # Default to ResNet
    input_channel = 512 if layers in [18,34] else 2048
    self.backbone = build_resnet(layers, pretrained=pretrained)
```

#### Updated `__main__` section:
- Added parameter count printing for verification

**Reason:** Enable MobileNetV3-Small as backbone while maintaining compatibility with ResNet.

**Key Technical Detail:**
- MobileNetV3-Small outputs 576 channels (vs ResNet-18's 512)
- Uses `mobilenet.features` which excludes the classifier head
- Feature map size is 7×7 at 224×224 input (same as ResNet)

---

### 3. `test.py`
**Major Changes:**

#### Added `benchmark_model()` function:
Comprehensive benchmarking that measures:

1. **Parameters Count**
   ```python
   total_params = sum(p.numel() for p in model.parameters())
   trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
   ```

2. **FLOPs (Floating Point Operations)**
   ```python
   from thop import profile
   flops, params = profile(model, inputs=(dummy_img, dummy_sal), verbose=False)
   ```

3. **GPU Memory Usage**
   ```python
   torch.cuda.reset_peak_memory_stats()
   _ = model(dummy_img, dummy_sal)
   gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
   ```

4. **Inference Speed**
   ```python
   # Warmup: 20 iterations
   # Measurement: 100 iterations
   fps = 100 / (end - start)
   avg_time = (end - start) / 100 * 1000  # ms
   ```

#### Modified `__main__` section:
- Calls `benchmark_model()` before evaluation
- Added graceful handling for missing weight file
- Added import for `os` module

**Reason:** Provide comprehensive performance metrics for research benchmarking.

---

### 4. `requirements.txt`
**Changes:**
- Added `thop` library for FLOPs calculation

**Reason:** Enable FLOPs profiling (optional dependency, gracefully skipped if unavailable).

---

## Unchanged Files

### `cadb_dataset.py`
**Status:** No modifications needed
**Reason:** Dataset loading is backbone-agnostic

### `samp_module.py`
**Status:** No modifications needed
**Reason:** Multi-pattern pooling modules work with any feature map dimensions

### `train.py`
**Status:** No modifications needed
**Reason:** Training loop automatically uses config settings

---

## Architecture Comparison

### Backbone Output Dimensions
| Backbone | Output Channels | Feature Map Size |
|----------|----------------|------------------|
| ResNet-18 | 512 | 7×7 |
| ResNet-50 | 2048 | 7×7 |
| MobileNetV3-Small | **576** | 7×7 |

### Expected Performance Trade-offs
| Metric | ResNet-18 | MobileNetV3-Small |
|--------|-----------|-------------------|
| Parameters | ~11M | ~2.5M (with SAMP) |
| Speed | Baseline | **2-3x faster** |
| Accuracy | Baseline | -1% to -3% (typical) |
| Memory | Baseline | **40-50% less** |

---

## Running the Code

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt --break-system-packages
```

### Step 2: Train the Model
```bash
python train.py
```

### Step 3: Test and Benchmark
```bash
python test.py
```

### Expected Benchmark Output
```
============================================================
BENCHMARK METRICS
============================================================

#Parameters: 2,XXX,XXX
Trainable Parameters: 2,XXX,XXX

FLOPs: XX,XXX,XXX (X.XX GFLOPs)
Params (thop): 2,XXX,XXX

GPU Memory (MB): XX.XX

Inference FPS: XX.XX
Average Inference Time: XX.XX ms
============================================================

Evaluation begining...
100%|████████████████████████| XXX/XXX [XX:XX<XX:XX, XX.XXit/s]

Evaluation result...
Test on XXXX images, Accuracy=XX.XX%, EMD(r=1)=X.XXXX, EMD(r=2)=X.XXXX, MSE_loss=X.XXXX, SRCC=X.XXXX, LCC=X.XXXX
```

---

## Technical Notes

### MobileNetV3-Small Architecture
- Based on inverted residual blocks with squeeze-excitation
- Uses h-swish activation function
- Optimized for mobile and edge devices
- Pre-trained on ImageNet available through torchvision

### Compatibility
- The code maintains full backward compatibility with ResNet
- To switch back to ResNet, change `backbone_type` in `config.py`
- All other modules (SAMP, attribute prediction, etc.) work identically

### Known Limitations
- `thop` library may not perfectly count custom operations in SAMP modules
- FLOPs reported are approximations
- Benchmark FPS is for single-image inference (batch=1)

---

## Benchmark Metrics to Report

For your research paper/benchmark, report these metrics:

1. **Model Complexity**
   - Total Parameters
   - FLOPs (GFLOPs)

2. **Efficiency**
   - Inference FPS
   - GPU Memory Usage (MB)

3. **Accuracy Metrics**
   - Accuracy (%)
   - EMD (r=1 and r=2)
   - MSE Loss
   - SRCC (Spearman Rank Correlation)
   - LCC (Linear Correlation Coefficient)

---

## Files Modified
✓ `config.py` - Updated hyperparameters and backbone type
✓ `samp_net.py` - Added MobileNetV3-Small support
✓ `test.py` - Added comprehensive benchmarking
✓ `requirements.txt` - Added thop dependency

## Files Unchanged
✓ `cadb_dataset.py` - No changes needed
✓ `samp_module.py` - No changes needed
✓ `train.py` - No changes needed (uses config)

---

## Questions?

If you encounter issues:
1. Check that dataset path exists: `/workspace/composition/CADB_Dataset`
2. Verify CUDA is available: `torch.cuda.is_available()`
3. Ensure sufficient GPU memory (reduce batch_size if OOM)
4. Check that all JSON files are in the dataset directory
