from samp_net import EMDLoss, SAMPNet
from cadb_dataset import CADBDataset
import torch
from torch.utils.data import DataLoader
import scipy.stats as stats
import numpy as np
from tqdm import tqdm
from config import Config
import time
import os

def calculate_accuracy(predict, target, threhold=2.6):
    assert target.shape == predict.shape, '{} vs. {}'.format(target.shape, predict.shape)
    bin_tar = target > threhold
    bin_pre = predict > threhold
    correct = (bin_tar == bin_pre).sum()
    acc     = correct.float() / target.size(0)
    return correct,acc

def calculate_lcc(target, predict):
    if len(target.shape) > 1:
        target = target.view(-1)
    if len(predict.shape) > 1:
        predict = predict.view(-1)
    predict = predict.cpu().numpy()
    target  = target.cpu().numpy()
    lcc = np.corrcoef(predict, target)[0,1]
    return lcc

def calculate_spearmanr(target, predict):
    if len(target.shape) > 1:
        target = target.view(-1)
    if len(predict.shape) > 1:
        predict = predict.view(-1)
    target_list = target.cpu().numpy().tolist()
    predict_list = predict.cpu().numpy().tolist()
    rho,_ = stats.spearmanr(predict_list, target_list)
    return rho

def dist2ave(pred_dist):
    pred_score = torch.sum(pred_dist* torch.Tensor(range(1,6)).to(pred_dist.device), dim=-1, keepdim=True)
    return pred_score

def benchmark_model(model, cfg):
    """Benchmark model performance metrics"""
    print("\n" + "="*60)
    print("BENCHMARK METRICS")
    print("="*60)
    
    device = next(model.parameters()).device
    model.eval()
    
    # Create dummy inputs
    dummy_img = torch.randn(1, 3, 224, 224).to(device)
    dummy_sal = torch.randn(1, 1, 224, 224).to(device)
    
    # 1. Parameter Count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n#Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # 2. FLOPs calculation
    try:
        from thop import profile
        import copy
        # Create a deep copy to avoid polluting the original model
        model_copy = copy.deepcopy(model)
        flops, params = profile(model_copy, inputs=(dummy_img.clone(), dummy_sal.clone()), verbose=False)
        print(f"\nFLOPs: {flops:,} ({flops/1e9:.2f} GFLOPs)")
        print(f"Params (thop): {params:,}")
        # Delete the copy to free memory
        del model_copy
    except ImportError:
        print("\nWarning: 'thop' not installed. Install with: pip install thop --break-system-packages")
        print("FLOPs calculation skipped.")
    except Exception as e:
        print(f"\nWarning: FLOPs calculation failed: {e}")
    
    # 3. GPU Memory Usage
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(dummy_img, dummy_sal)
        gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"\nGPU Memory (MB): {gpu_memory_mb:.2f}")
    
    # 4. Inference Speed (FPS)
    print("\nMeasuring inference speed...")
    with torch.no_grad():
        # Warmup
        for _ in range(20):
            _ = model(dummy_img, dummy_sal)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(100):
            _ = model(dummy_img, dummy_sal)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end = time.time()
        fps = 100 / (end - start)
        avg_time = (end - start) / 100 * 1000  # Convert to ms
        print(f"Inference FPS: {fps:.2f}")
        print(f"Average Inference Time: {avg_time:.2f} ms")
    
    print("="*60 + "\n")

def evaluation_on_cadb(model, cfg):
    model.eval()
    device = next(model.parameters()).device
    testdataset = CADBDataset('test', cfg)
    testloader = DataLoader(testdataset,
                            batch_size=cfg.batch_size,
                            shuffle=False,
                            num_workers=cfg.num_workers,
                            drop_last=False)
    emd_r2_fn = EMDLoss(reduction='sum', r=2)
    emd_r1_fn = EMDLoss(reduction='sum', r=1)
    emd_r2_error = 0.0
    emd_r1_error = 0.0
    correct   = 0.
    tar_scores = None
    pre_scores = None
    print()
    print('Evaluation begining...')
    with torch.no_grad():
        for (im,score,dist,saliency,attributes) in tqdm(testloader):
            image = im.to(device)
            score = score.to(device)
            dist  = dist.to(device)
            saliency = saliency.to(device)
            weight, atts, output = model(image, saliency)

            pred_score = dist2ave(output)
            emd_r1_error += emd_r1_fn(dist, output).item()
            emd_r2_error += emd_r2_fn(dist, output).item()
            correct   += calculate_accuracy(pred_score, score)[0].item()
            if tar_scores is None:
                tar_scores = score
                pre_scores = pred_score
            else:
                tar_scores = torch.cat([tar_scores, score], dim=0)
                pre_scores = torch.cat([pre_scores, pred_score], dim=0)
    print('Evaluation result...')
    # print('Scores shape', pre_scores.shape, tar_scores.shape)
    avg_mse = torch.nn.MSELoss()(pre_scores.view(-1), tar_scores.view(-1)).item()
    SRCC = calculate_spearmanr(tar_scores, pre_scores)
    LCC  = calculate_lcc(tar_scores, pre_scores)
    avg_r1_emd = emd_r1_error / len(testdataset)
    avg_r2_emd = emd_r2_error / len(testdataset)
    avg_acc = correct / len(testdataset)
    ss = "Test on {} images, Accuracy={:.2%}, EMD(r=1)={:.4f}, EMD(r=2)={:.4f},". \
        format(len(testdataset), avg_acc, avg_r1_emd, avg_r2_emd)
    ss += " MSE_loss={:.4f}, SRCC={:.4f}, LCC={:.4f}". \
        format(avg_mse, SRCC, LCC)
    print(ss)
    return avg_acc, avg_r1_emd, avg_r2_emd, avg_mse, SRCC, LCC

if __name__ == '__main__':
    cfg = Config()
    device = torch.device('cuda:{}'.format(cfg.gpu_id))
    model = SAMPNet(cfg, pretrained=False).to(device)
    
    # Note: Pretrained weights from ResNet-based SAMP-Net are NOT compatible with MobileNetV3
    # You need to train from scratch or use pretrained MobileNetV3 backbone (handled in __init__)
    weight_file = 'C:/Users/segov/Music/SAMP-Net/SAMPNet/experiments/mobilenetv3_small_samp_aaff_wemd/checkpoints/model-best.pth'
    
    if os.path.exists(weight_file):
        try:
            model.load_state_dict(torch.load(weight_file, map_location=device))
            print(f"✓ Loaded weights from {weight_file}")
        except Exception as e:
            print(f"✗ Failed to load weights: {e}")
            print("Using ImageNet pretrained MobileNetV3 backbone with random task-specific layers.")
    else:
        print(f"Weight file {weight_file} not found.")
        print("Using ImageNet pretrained MobileNetV3 backbone with random task-specific layers.")
        print("You need to train the model first using: python train.py")
    
    # Run benchmarks
    print("\nRunning benchmarks...")
    benchmark_model(model, cfg)
    
    # Only run evaluation if weights are loaded
    if os.path.exists(weight_file):
        print("\nRunning evaluation on test set...")
        evaluation_on_cadb(model, cfg)
    else:
        print("\nSkipping evaluation (no trained weights found).")
        print("Train the model first: python train.py")
