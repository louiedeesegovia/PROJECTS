# SAMP-Net with CSPNet Backbone

This is a modified version of SAMP-Net (Saliency-Aware Multi-Pattern Pooling Network) that uses CSPNet (Cross-Stage Partial Networks) as the backbone instead of ResNet.

## Changes from Original Code

### 1. **Backbone Architecture**
- **Original**: ResNet-18/34/50/101
- **Modified**: CSPDarkNet53 or CSPResNet50
- CSPNet provides better gradient flow and reduced computational complexity through cross-stage partial connections

### 2. **Configuration Parameters** (config.py)
Modified parameters as requested:
- `batch_size = 4` (changed from 16)
- `max_epoch = 50` (unchanged)
- `lr = 1e-4` (unchanged)
- `num_workers = 0` (changed from 8)

New parameters:
- `cspnet_variant = 'cspdarknet53'` - Options: 'cspdarknet53' or 'cspresnet50'

### 3. **New Files**
- `cspnet_backbone.py`: Implementation of CSPNet architectures
  - CSPDarkNet53: 1024 output channels
  - CSPResNet50: 2048 output channels

### 4. **Modified Files**
- `config.py`: Updated with CSPNet-specific configurations and benchmark parameters
- `samp_net.py`: Modified to use CSPNet backbone instead of ResNet
- Other files remain unchanged: `samp_module.py`, `cadb_dataset.py`, `train.py`, `test.py`

## CSPNet Variants

### CSPDarkNet53
- Lighter model with 1024 output channels
- Better for faster training and inference
- ~7x7 feature maps at output
- Uses Mish activation function

### CSPResNet50
- Deeper model with 2048 output channels
- More parameters but potentially better performance
- ~7x7 feature maps at output
- Uses Mish activation function

## Installation

```bash
pip install -r requirements.txt
```

## Directory Structure

Ensure your dataset is organized as follows:
```
/workspace/composition/CADB_Dataset/
├── images/
├── composition_scores.json
├── composition_attributes.json
├── emdloss_weight.json
└── split.json
```

## Usage

### Training

```bash
python train.py
```

The model will:
- Use CSPDarkNet53 backbone by default
- Train for 50 epochs with batch size 4
- Save checkpoints every epoch to `./experiments/cspnet_cspdarknet53_samp_aaff_wemd/checkpoints/`
- Log training progress to TensorBoard

### Testing

```bash
python test.py
```

Make sure to update the `weight_file` path in `test.py` to point to your trained model.

### Changing CSPNet Variant

To switch between CSPNet variants, modify `config.py`:

```python
# For CSPDarkNet53 (default)
cspnet_variant = 'cspdarknet53'

# For CSPResNet50
cspnet_variant = 'cspresnet50'
```

## Model Architecture

```
Input (3, 224, 224)
    ↓
CSPNet Backbone
    ↓
Feature Maps (1024/2048, 7, 7)
    ↓
SAMP Module (Saliency-Aware Multi-Pattern Pooling)
    ↓
Attribute Branch + Composition Branch
    ↓
Score Distribution (5 classes)
```

## Key Features Retained

- ✅ Saliency-aware pooling
- ✅ Multi-pattern pooling
- ✅ Pattern weighting
- ✅ Attribute prediction
- ✅ Channel attention (AAFF)
- ✅ Weighted EMD loss

## Benchmark Configuration

The configuration is set for benchmarking with:
- Small batch size (4) for memory efficiency
- Single worker for debugging
- 50 epochs for fair comparison
- Same learning rate and optimization strategy

## Expected Output

During training, you'll see:
```
Experiment name cspnet_cspdarknet53_samp_aaff_wemd

Create experiment directory: ./experiments/cspnet_cspdarknet53_samp_aaff_wemd
Training Epoch:0/50 Current Batch: 10/X EMD_Loss:X.XXXX Attribute_Loss:X.XXXX ACC:XX.XX% lr:0.000100
...
```

## Evaluation Metrics

The model is evaluated on:
- **Accuracy**: Binary classification accuracy (threshold=2.6)
- **EMD (r=1)**: Earth Mover's Distance with r=1
- **EMD (r=2)**: Earth Mover's Distance with r=2
- **MSE**: Mean Squared Error
- **SRCC**: Spearman Rank Correlation Coefficient
- **LCC**: Linear Correlation Coefficient

## Notes

1. **Pretrained Weights**: CSPNet doesn't have ImageNet pretrained weights by default. The model trains from scratch unless you manually load pretrained weights.

2. **Memory Usage**: CSPDarkNet53 uses less memory than CSPResNet50. If you encounter OOM errors with batch_size=4, you can:
   - Use CSPDarkNet53 instead of CSPResNet50
   - Reduce batch size to 2
   - Reduce image size in config

3. **Performance**: CSPNet is designed to be more efficient than ResNet while maintaining comparable or better performance. Expect:
   - Faster training per epoch
   - Similar or better gradient flow
   - Potentially different convergence behavior

## Troubleshooting

**Issue**: CUDA out of memory
**Solution**: 
```python
# In config.py
batch_size = 2  # Reduce from 4
# or
cspnet_variant = 'cspdarknet53'  # Use lighter variant
```

**Issue**: Dataset not found
**Solution**: Update the dataset_path in config.py:
```python
dataset_path = '/your/path/to/CADB_Dataset'
```

## Citation

If you use this code in your research, please cite the original SAMP-Net paper and CSPNet:

```bibtex
@article{cspnet,
  title={CSPNet: A new backbone that can enhance learning capability of CNN},
  author={Wang, Chien-Yao and Liao, Hong-Yuan Mark and Wu, Yeh-Hua and Chen, Ping-Yang and Hsieh, Jun-Wei and Yeh, I-Hau},
  journal={CVPRW},
  year={2020}
}
```

## License

Please refer to the original repository's license for usage terms.
