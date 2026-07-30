"""
Comprehensive Benchmark Summary Script for CSPNet-SAMP

This script provides a complete benchmark report including:
- Model parameters and size
- FLOPs (computational complexity)
- Inference speed (FPS)
- GPU memory usage
- Training time statistics

Run this after training to get a complete benchmark report.
"""

import torch
import time
import numpy as np
from config import Config
from samp_net import SAMPNet
import json
import os

try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: thop not installed. Install with: pip install thop")


def count_parameters(model):
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def measure_flops(model, device):
    """Measure FLOPs using thop"""
    if not THOP_AVAILABLE:
        return None, None
    
    model.eval()
    dummy_image = torch.randn(1, 3, 224, 224).to(device)
    dummy_saliency = torch.randn(1, 1, 224, 224).to(device)
    
    try:
        flops, params = profile(model, inputs=(dummy_image, dummy_saliency), verbose=False)
        return flops, params
    except Exception as e:
        print(f"Error measuring FLOPs: {e}")
        return None, None


def measure_inference_speed(model, device, batch_size=1, num_warmup=20, num_iterations=100):
    """Measure inference speed (FPS)"""
    model.eval()
    dummy_image = torch.randn(batch_size, 3, 224, 224).to(device)
    dummy_saliency = torch.randn(batch_size, 1, 224, 224).to(device)
    
    with torch.no_grad():
        # Warmup
        for _ in range(num_warmup):
            _ = model(dummy_image, dummy_saliency)
        
        # Measure
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iterations):
            _ = model(dummy_image, dummy_saliency)
        torch.cuda.synchronize()
        end = time.time()
    
    total_time = end - start
    avg_time = total_time / num_iterations
    fps = (num_iterations * batch_size) / total_time
    
    return fps, avg_time * 1000  # Return FPS and time in ms


def measure_memory(model, device, batch_size=1):
    """Measure GPU memory usage"""
    model.eval()
    dummy_image = torch.randn(batch_size, 3, 224, 224).to(device)
    dummy_saliency = torch.randn(batch_size, 1, 224, 224).to(device)
    
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(dummy_image, dummy_saliency)
    
    memory_allocated = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
    memory_reserved = torch.cuda.memory_reserved(device) / 1024**2  # MB
    
    return memory_allocated, memory_reserved


def parse_training_log(log_dir):
    """Parse training logs to extract timing information"""
    # This is a placeholder - implement based on your log format
    # For now, return dummy data
    return {
        'total_time_hours': 0.0,
        'avg_epoch_time_min': 0.0,
        'epochs_completed': 0
    }


def generate_benchmark_report(cfg, checkpoint_path=None):
    """Generate comprehensive benchmark report"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE BENCHMARK REPORT - CSPNet-SAMP")
    print("="*80)
    
    # Setup
    device = torch.device(f'cuda:{cfg.gpu_id}' if torch.cuda.is_available() else 'cpu')
    model = SAMPNet(cfg, pretrained=False).to(device)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\nLoading checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))
    
    benchmark_results = {}
    
    # 1. Model Configuration
    print("\n" + "-"*80)
    print("1. MODEL CONFIGURATION")
    print("-"*80)
    print(f"Backbone: {cfg.cspnet_variant}")
    print(f"Image Size: {cfg.image_size}")
    print(f"Batch Size: {cfg.batch_size}")
    print(f"Learning Rate: {cfg.lr}")
    print(f"Max Epochs: {cfg.max_epoch}")
    
    benchmark_results['config'] = {
        'backbone': cfg.cspnet_variant,
        'image_size': cfg.image_size,
        'batch_size': cfg.batch_size,
        'learning_rate': cfg.lr,
        'max_epochs': cfg.max_epoch
    }
    
    # 2. Model Size
    print("\n" + "-"*80)
    print("2. MODEL SIZE")
    print("-"*80)
    total_params, trainable_params = count_parameters(model)
    model_size_mb = total_params * 4 / 1024 / 1024
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Model Size: {model_size_mb:.2f} MB")
    
    benchmark_results['model_size'] = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'size_mb': model_size_mb
    }
    
    # 3. Computational Complexity (FLOPs)
    print("\n" + "-"*80)
    print("3. COMPUTATIONAL COMPLEXITY")
    print("-"*80)
    flops, _ = measure_flops(model, device)
    if flops is not None:
        print(f"FLOPs: {flops:,} ({flops/1e9:.2f} GFLOPs)")
        benchmark_results['flops'] = {
            'total_flops': flops,
            'gflops': flops/1e9
        }
    else:
        print("FLOPs: Not available")
        benchmark_results['flops'] = None
    
    # 4. Inference Speed
    print("\n" + "-"*80)
    print("4. INFERENCE SPEED")
    print("-"*80)
    
    # Single image
    fps_single, time_single = measure_inference_speed(model, device, batch_size=1)
    print(f"Single Image:")
    print(f"  Inference Time: {time_single:.2f} ms")
    print(f"  FPS: {fps_single:.2f} images/sec")
    
    # Batch processing
    fps_batch, time_batch = measure_inference_speed(model, device, batch_size=cfg.batch_size)
    print(f"\nBatch Processing (batch_size={cfg.batch_size}):")
    print(f"  Batch Time: {time_batch:.2f} ms")
    print(f"  Throughput: {fps_batch:.2f} images/sec")
    
    benchmark_results['inference_speed'] = {
        'single_image': {
            'time_ms': time_single,
            'fps': fps_single
        },
        'batch': {
            'batch_size': cfg.batch_size,
            'time_ms': time_batch,
            'throughput': fps_batch
        }
    }
    
    # 5. Memory Usage
    print("\n" + "-"*80)
    print("5. GPU MEMORY USAGE")
    print("-"*80)
    
    # Single image
    mem_alloc_single, mem_reserved_single = measure_memory(model, device, batch_size=1)
    print(f"Single Image:")
    print(f"  Allocated: {mem_alloc_single:.2f} MB")
    print(f"  Reserved: {mem_reserved_single:.2f} MB")
    
    # Batch
    mem_alloc_batch, mem_reserved_batch = measure_memory(model, device, batch_size=cfg.batch_size)
    print(f"\nBatch Processing (batch_size={cfg.batch_size}):")
    print(f"  Allocated: {mem_alloc_batch:.2f} MB")
    print(f"  Reserved: {mem_reserved_batch:.2f} MB")
    
    benchmark_results['memory'] = {
        'single_image': {
            'allocated_mb': mem_alloc_single,
            'reserved_mb': mem_reserved_single
        },
        'batch': {
            'batch_size': cfg.batch_size,
            'allocated_mb': mem_alloc_batch,
            'reserved_mb': mem_reserved_batch
        }
    }
    
    # 6. Training Statistics (if available)
    print("\n" + "-"*80)
    print("6. TRAINING STATISTICS")
    print("-"*80)
    
    csv_path = os.path.join(cfg.exp_path, '..', f'{cfg.exp_name}.csv')
    if os.path.exists(csv_path):
        print(f"Reading training statistics from: {csv_path}")
        # You can parse the CSV here to get training time info
        print("See CSV file for detailed training metrics")
    else:
        print("Training statistics not available (model not trained yet)")
    
    # 7. Summary Table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Metric':<40} {'Value':>35}")
    print("-"*80)
    print(f"{'Backbone':<40} {cfg.cspnet_variant:>35}")
    print(f"{'Total Parameters':<40} {total_params:>35,}")
    print(f"{'Model Size (MB)':<40} {model_size_mb:>35.2f}")
    if flops:
        print(f"{'FLOPs (GFLOPs)':<40} {flops/1e9:>35.2f}")
    print(f"{'Inference Time - Single (ms)':<40} {time_single:>35.2f}")
    print(f"{'FPS - Single':<40} {fps_single:>35.2f}")
    print(f"{'Inference Time - Batch (ms)':<40} {time_batch:>35.2f}")
    print(f"{'Throughput - Batch (img/s)':<40} {fps_batch:>35.2f}")
    print(f"{'GPU Memory - Single (MB)':<40} {mem_alloc_single:>35.2f}")
    print(f"{'GPU Memory - Batch (MB)':<40} {mem_alloc_batch:>35.2f}")
    print("="*80 + "\n")
    
    # Save to JSON
    output_path = os.path.join(cfg.exp_path, 'benchmark_results.json')
    os.makedirs(cfg.exp_path, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(benchmark_results, f, indent=4)
    print(f"Benchmark results saved to: {output_path}\n")
    
    return benchmark_results


if __name__ == '__main__':
    cfg = Config()
    
    # Option 1: Benchmark without trained weights
    print("Benchmarking model architecture (no trained weights)...")
    results = generate_benchmark_report(cfg)
    
    # Option 2: Benchmark with trained weights
    # Uncomment and modify path as needed:
    # checkpoint_path = './experiments/cspnet_cspdarknet53_samp_aaff_wemd/checkpoints/model-best.pth'
    # results = generate_benchmark_report(cfg, checkpoint_path)
