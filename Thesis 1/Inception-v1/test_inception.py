#!/usr/bin/env python
"""
Quick test script to verify Inception-v1 backbone integration
This tests the model can be created and performs a forward pass
"""

import torch
from config import Config
from samp_net import SAMPNet

def test_inception_backbone():
    print("="*60)
    print("Testing Inception-v1 Backbone Integration")
    print("="*60)
    
    # Create config
    cfg = Config()
    print(f"\nConfiguration:")
    print(f"  Backbone: {cfg.backbone_type}")
    print(f"  Batch Size: {cfg.batch_size}")
    print(f"  Image Size: {cfg.image_size}")
    print(f"  Learning Rate: {cfg.lr}")
    print(f"  Max Epochs: {cfg.max_epoch}")
    
    # Create model
    print("\nCreating model...")
    model = SAMPNet(cfg, pretrained=False)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)
    s = torch.randn(batch_size, 1, 224, 224)
    
    model.eval()
    with torch.no_grad():
        weight, attribute, score = model(x, s)
    
    print(f"\nOutput shapes:")
    if weight is not None:
        print(f"  Weight: {weight.shape}")
    if attribute is not None:
        print(f"  Attribute: {attribute.shape}")
    print(f"  Score: {score.shape}")
    
    # Verify output shapes
    assert score.shape == (batch_size, 5), f"Expected score shape ({batch_size}, 5), got {score.shape}"
    if attribute is not None:
        assert attribute.shape == (batch_size, cfg.num_attributes), \
            f"Expected attribute shape ({batch_size}, {cfg.num_attributes}), got {attribute.shape}"
    
    # Verify score is a probability distribution
    score_sum = score.sum(dim=1)
    assert torch.allclose(score_sum, torch.ones(batch_size), atol=1e-5), \
        f"Score should sum to 1, got {score_sum}"
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)
    
    return model

def test_with_cuda():
    """Test if CUDA is available and run on GPU"""
    if torch.cuda.is_available():
        print("\n" + "="*60)
        print("Testing on CUDA")
        print("="*60)
        
        device = torch.device('cuda:0')
        cfg = Config()
        model = SAMPNet(cfg, pretrained=False).to(device)
        
        x = torch.randn(2, 3, 224, 224).to(device)
        s = torch.randn(2, 1, 224, 224).to(device)
        
        model.eval()
        with torch.no_grad():
            weight, attribute, score = model(x, s)
        
        print(f"✓ CUDA forward pass successful")
        print(f"  Device: {device}")
        print(f"  Score device: {score.device}")
    else:
        print("\nCUDA not available, skipping GPU test")

if __name__ == '__main__':
    # Test CPU
    model = test_inception_backbone()
    
    # Test CUDA if available
    test_with_cuda()
    
    print("\n✓ Inception-v1 backbone is working correctly!")
