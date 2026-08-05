# SAMP-Net with MobileNetV3-Small Backbone

## Modifications Made

### 1. **Backbone Changed to MobileNetV3-Small**
   - Original: ResNet-18/34/50/101
   - New: MobileNetV3-Small (576 output channels)
   - Location: `samp_net.py` - `build_mobilenetv3_small()` function

### 2. **Configuration Updates** (`config.py`)
   - `batch_size = 4` (reduced from 16)
   - `max_epoch = 50` (kept as requested)
   - `lr = 1e-4` (kept as requested)
   - `num_workers = 0` (reduced from 8)
   - `backbone_type = 'mobilenetv3_small'` (new parameter)

### 3. **Benchmarking Added** (`test.py`)
   The test script now includes comprehensive benchmarking:
   - **Parameters Count**: Total and trainable parameters
   - **FLOPs**: Floating point operations (requires `thop` library)
   - **GPU Memory Usage**: Peak memory allocation in MB
   - **Inference Speed**: FPS and average inference time in ms

## Installation

```bash
pip install -r requirements.txt --break-system-packages
```

Note: If `thop` installation fails, FLOPs calculation will be skipped but other metrics will still work.

## Usage

### Training
```bash
python train.py
```

### Testing with Benchmarks
```bash
python test.py
```

The test script will output:
1. Benchmark metrics (Parameters, FLOPs, GPU Memory, FPS)
2. Evaluation metrics (Accuracy, EMD, MSE, SRCC, LCC)

## Expected Outputs

### Benchmark Metrics
```
============================================================
BENCHMARK METRICS
============================================================

#Parameters: X,XXX,XXX
Trainable Parameters: X,XXX,XXX

FLOPs: XXX,XXX,XXX (X.XX GFLOPs)
Params (thop): X,XXX,XXX

GPU Memory (MB): XX.XX

Inference FPS: XX.XX
Average Inference Time: XX.XX ms
============================================================
```

### Evaluation Metrics
```
Test on XXXX images, Accuracy=XX.XX%, EMD(r=1)=X.XXXX, EMD(r=2)=X.XXXX, MSE_loss=X.XXXX, SRCC=X.XXXX, LCC=X.XXXX
```

## Key Differences: MobileNetV3-Small vs ResNet-18

| Metric | ResNet-18 | MobileNetV3-Small |
|--------|-----------|-------------------|
| Parameters | ~11M | ~1.5-2.5M (estimated with SAMP modules) |
| FLOPs | ~1.8 GFLOPs | ~0.06 GFLOPs (backbone only) |
| Output Channels | 512 | 576 |
| Speed | Slower | Faster |
| Accuracy | Higher | Slightly lower (typical trade-off) |

## Architecture Details

The modified network maintains the same overall structure:
1. **Backbone**: MobileNetV3-Small (replaces ResNet)
2. **SAMP Module**: Saliency-Aware Multi-Pattern Pooling (unchanged)
3. **Attribute Branch**: Multi-task learning for composition attributes (unchanged)
4. **Composition Prediction**: EMD loss for score distribution (unchanged)

## Training Tips for MobileNetV3-Small

1. **Learning Rate**: Keep at 1e-4 or slightly higher (1.5e-4) since MobileNet is lighter
2. **Batch Size**: 4 is good for limited GPU memory; can increase to 8 or 16 if memory allows
3. **Data Augmentation**: May need more aggressive augmentation than ResNet due to smaller capacity
4. **Warmup**: Consider adding learning rate warmup for first few epochs
5. **Fine-tuning**: The backbone learning rate is already set to 0.01x the main learning rate

## Dataset Structure

Ensure your dataset is organized as expected:
```
/workspace/composition/CADB_Dataset/
├── images/
├── composition_scores.json
├── split.json
├── composition_attributes.json
└── emdloss_weight.json
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in `config.py` (try 2 or 1)
- Reduce `image_size` (try 192 instead of 224)

### Slow Training
- Increase `num_workers` if you have sufficient CPU cores
- Use mixed precision training (requires code modification)

### Poor Accuracy
- Check if pretrained=True in SAMPNet initialization
- Verify dataset paths and file integrity
- Monitor training curves in TensorBoard

## Citation

If you use this code for your research, please cite the original SAMP-Net paper and acknowledge the MobileNetV3 modification.
