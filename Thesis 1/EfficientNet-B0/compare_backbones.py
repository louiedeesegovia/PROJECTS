"""
Comparison script for benchmarking different backbones
"""
import torch
import time
from config import Config
from samp_net import SAMPNet

def benchmark_backbone(backbone_type, device='cpu'):
    """Benchmark a specific backbone"""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {backbone_type}")
    print(f"{'='*60}")
    
    # Create config with specific backbone
    cfg = Config()
    cfg.backbone_type = backbone_type
    
    # Create model
    model = SAMPNet(cfg, pretrained=False)
    model = model.to(device)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Prepare dummy inputs
    dummy_image = torch.randn(1, 3, 224, 224).to(device)
    dummy_saliency = torch.randn(1, 1, 224, 224).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_image, dummy_saliency)
    
    # Measure inference time
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_image, dummy_saliency)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    end = time.time()
    fps = 100 / (end - start)
    
    # Calculate FLOPs if thop is available
    try:
        from thop import profile
        flops, _ = profile(model, inputs=(dummy_image, dummy_saliency), verbose=False)
        flops_gflops = flops / 1e9
    except:
        flops_gflops = None
    
    # Memory usage
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(dummy_image, dummy_saliency)
        memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        memory_mb = None
    
    # Print results
    print(f"\nResults:")
    print(f"  Parameters (M):  {total_params/1e6:.2f}")
    if flops_gflops is not None:
        print(f"  FLOPs (G):       {flops_gflops:.2f}")
    print(f"  Inference FPS:   {fps:.2f}")
    if memory_mb is not None:
        print(f"  GPU Memory (MB): {memory_mb:.2f}")
    
    return {
        'backbone': backbone_type,
        'params_m': total_params / 1e6,
        'flops_g': flops_gflops,
        'fps': fps,
        'memory_mb': memory_mb
    }

def compare_backbones(backbones=['efficientnet_b0', 'resnet18'], device='cpu'):
    """Compare multiple backbones"""
    print(f"\n{'='*60}")
    print(f"BACKBONE COMPARISON")
    print(f"Device: {device}")
    print(f"{'='*60}")
    
    results = []
    for backbone in backbones:
        try:
            result = benchmark_backbone(backbone, device)
            results.append(result)
        except Exception as e:
            print(f"\nError with {backbone}: {e}")
    
    # Print comparison table
    if results:
        print(f"\n{'='*60}")
        print(f"COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"{'Backbone':<20} {'Params(M)':<12} {'FLOPs(G)':<12} {'FPS':<10} {'Mem(MB)':<10}")
        print(f"{'-'*60}")
        for r in results:
            params = f"{r['params_m']:.2f}"
            flops = f"{r['flops_g']:.2f}" if r['flops_g'] is not None else "N/A"
            fps = f"{r['fps']:.2f}"
            mem = f"{r['memory_mb']:.2f}" if r['memory_mb'] is not None else "N/A"
            print(f"{r['backbone']:<20} {params:<12} {flops:<12} {fps:<10} {mem:<10}")
        print(f"{'='*60}\n")
    
    return results

if __name__ == '__main__':
    import sys
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cpu':
        print("Warning: Running on CPU. For GPU benchmarks, ensure CUDA is available.")
    
    # Define backbones to compare
    backbones = ['efficientnet_b0', 'resnet18']
    
    # You can add more backbones if needed:
    # backbones = ['efficientnet_b0', 'resnet18', 'resnet34', 'resnet50']
    
    # Run comparison
    results = compare_backbones(backbones, device)
    
    print("\n✓ Comparison complete!")
