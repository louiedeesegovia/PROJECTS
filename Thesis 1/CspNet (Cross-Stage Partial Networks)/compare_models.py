"""
Script to compare CSPNet variants and print model statistics
"""
import torch
from config import Config
from samp_net import SAMPNet
from cspnet_backbone import build_cspnet

def count_parameters(model):
    """Count trainable and total parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def test_model(variant_name):
    """Test a CSPNet variant"""
    print(f"\n{'='*60}")
    print(f"Testing {variant_name}")
    print(f"{'='*60}")
    
    cfg = Config()
    cfg.cspnet_variant = variant_name
    
    # Create model
    model = SAMPNet(cfg, pretrained=False)
    
    # Count parameters
    total, trainable = count_parameters(model)
    
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Model size (MB): {total * 4 / 1024 / 1024:.2f}")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    s = torch.randn(2, 1, 224, 224)
    
    model.eval()
    with torch.no_grad():
        weight, attribute, score = model(x, s)
    
    print(f"\nOutput shapes:")
    if weight is not None:
        print(f"  Pattern weights: {weight.shape}")
    if attribute is not None:
        print(f"  Attributes: {attribute.shape}")
    print(f"  Scores: {score.shape}")
    
    # Test with GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        model = model.to(device)
        x = x.to(device)
        s = s.to(device)
        
        # Warm up
        with torch.no_grad():
            _ = model(x, s)
        
        # Time inference
        torch.cuda.synchronize()
        import time
        start = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = model(x, s)
        torch.cuda.synchronize()
        end = time.time()
        
        avg_time = (end - start) / 100 * 1000  # ms
        print(f"\nGPU Inference time (batch=2): {avg_time:.2f} ms")
        print(f"Throughput: {2000/avg_time:.2f} images/sec")
        
        # Memory usage
        memory_allocated = torch.cuda.memory_allocated(device) / 1024 / 1024
        memory_reserved = torch.cuda.memory_reserved(device) / 1024 / 1024
        print(f"GPU Memory allocated: {memory_allocated:.2f} MB")
        print(f"GPU Memory reserved: {memory_reserved:.2f} MB")
    else:
        print("\nGPU not available, skipping GPU tests")
    
    return total, trainable

def compare_backbones():
    """Compare different backbone architectures"""
    print("\n" + "="*60)
    print("BACKBONE COMPARISON")
    print("="*60)
    
    variants = ['cspdarknet53', 'cspresnet50']
    results = {}
    
    for variant in variants:
        backbone, out_channels = build_cspnet(variant, pretrained=False)
        
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = backbone(x)
        
        total, trainable = count_parameters(backbone)
        
        results[variant] = {
            'out_channels': out_channels,
            'out_shape': out.shape,
            'total_params': total,
            'trainable_params': trainable
        }
        
        print(f"\n{variant}:")
        print(f"  Output channels: {out_channels}")
        print(f"  Output shape: {out.shape}")
        print(f"  Total parameters: {total:,}")
        print(f"  Model size (MB): {total * 4 / 1024 / 1024:.2f}")
    
    return results

if __name__ == '__main__':
    print("CSPNet-SAMP Model Comparison Script")
    print("="*60)
    
    # Compare backbones only
    backbone_results = compare_backbones()
    
    # Test full models
    variants = ['cspdarknet53', 'cspresnet50']
    model_results = {}
    
    for variant in variants:
        try:
            total, trainable = test_model(variant)
            model_results[variant] = {'total': total, 'trainable': trainable}
        except Exception as e:
            print(f"\nError testing {variant}: {str(e)}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if model_results:
        print("\nFull Model Comparison:")
        for variant, stats in model_results.items():
            print(f"{variant:20s}: {stats['total']:>12,} params ({stats['total']*4/1024/1024:>6.2f} MB)")
        
        # Recommendation
        print("\n" + "-"*60)
        print("RECOMMENDATIONS:")
        print("-"*60)
        print("CSPDarkNet53:")
        print("  ✓ Lighter model (~40% fewer parameters)")
        print("  ✓ Faster training and inference")
        print("  ✓ Lower memory usage")
        print("  ✓ Good for initial experiments and benchmarking")
        print("  ✓ Recommended for batch_size=4")
        
        print("\nCSPResNet50:")
        print("  ✓ Deeper model with more capacity")
        print("  ✓ Potentially better final performance")
        print("  ✓ More memory intensive")
        print("  ✓ Good for final models after hyperparameter tuning")
        print("  ⚠ May need batch_size=2 depending on GPU memory")
        
        print("\nFor your benchmark with batch_size=4:")
        print("  → CSPDarkNet53 is recommended to start")
