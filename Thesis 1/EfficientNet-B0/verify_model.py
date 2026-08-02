"""
Quick verification script to test EfficientNet-B0 SAMP model
"""
import torch
from config import Config
from samp_net import SAMPNet

def test_model_creation():
    print("="*60)
    print("Testing EfficientNet-B0 SAMP Model Creation")
    print("="*60)
    
    # Create config
    cfg = Config()
    print(f"\nConfiguration:")
    print(f"  Backbone: {cfg.backbone_type}")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  Image size: {cfg.image_size}")
    print(f"  Learning rate: {cfg.lr}")
    print(f"  Max epochs: {cfg.max_epoch}")
    print(f"  Num workers: {cfg.num_workers}")
    
    # Create model
    print("\nCreating model...")
    model = SAMPNet(cfg, pretrained=False)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Parameters (M): {total_params/1e6:.2f}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    dummy_image = torch.randn(2, 3, 224, 224)
    dummy_saliency = torch.randn(2, 1, 224, 224)
    
    model.eval()
    with torch.no_grad():
        weight, attribute, score = model(dummy_image, dummy_saliency)
    
    print(f"\nOutput shapes:")
    if weight is not None:
        print(f"  Pattern weights: {weight.shape}")
    if attribute is not None:
        print(f"  Attributes: {attribute.shape}")
    print(f"  Scores: {score.shape}")
    
    # Verify output
    assert score.shape == (2, 5), f"Expected score shape (2, 5), got {score.shape}"
    assert torch.allclose(score.sum(dim=1), torch.ones(2)), "Scores should sum to 1 (softmax)"
    
    print("\n" + "="*60)
    print("✓ Model creation and forward pass successful!")
    print("="*60)
    
    return model

if __name__ == '__main__':
    try:
        model = test_model_creation()
        print("\n✓ All tests passed!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
