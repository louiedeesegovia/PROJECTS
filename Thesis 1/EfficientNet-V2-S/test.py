from samp_net import EMDLoss, SAMPNet
from cadb_dataset import CADBDataset
import torch
from torch.utils.data import DataLoader
import scipy.stats as stats
import numpy as np
from tqdm import tqdm
from config import Config
import time

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

def benchmark_model(model, device):
    """
    Benchmark model for FPS, FLOPs, Parameters, and Memory usage
    """
    print("\n" + "="*60)
    print("BENCHMARK METRICS - EfficientNet-V2-S")
    print("="*60)
    
    model.eval()
    dummy_image = torch.randn(1, 3, 224, 224).to(device)
    dummy_saliency = torch.randn(1, 1, 224, 224).to(device)
    
    # Warmup
    print("Warming up model...")
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_image, dummy_saliency)
    
    # FPS measurement
    print("Measuring inference speed...")
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_image, dummy_saliency)
    torch.cuda.synchronize()
    end = time.time()
    fps = 100 / (end - start)
    print(f"Inference FPS: {fps:.2f}")
    
    # FLOPs and Parameters
    print("Calculating FLOPs and Parameters...")
    try:
        from thop import profile
        flops, params = profile(model, inputs=(dummy_image, dummy_saliency), verbose=False)
        print(f"FLOPs: {flops:,} ({flops/1e9:.2f} GFLOPs)")
        print(f"Parameters: {params:,} ({params/1e6:.2f} M)")
    except ImportError:
        print("Warning: thop not installed. Run 'pip install thop' to measure FLOPs.")
        # Alternative parameter counting
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f} M)")
        print(f"Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f} M)")
    
    # Memory usage
    print("Measuring GPU memory usage...")
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(dummy_image, dummy_saliency)
    memory_mb = torch.cuda.max_memory_allocated() / 1024**2
    print(f"GPU Memory (MB): {memory_mb:.2f}")
    
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
    
    # Optional: Load pretrained weights if available
    weight_file = 'C:/Users/segov/Music/SAMP-Net/SAMPNet/experiments/efficientnet_v2_s_samp_aaff_wemd/checkpoints/model-best.pth'
    import os
    if os.path.exists(weight_file):
        model.load_state_dict(torch.load(weight_file))
        print(f"Loaded weights from {weight_file}")
    
    # Run benchmark
    benchmark_model(model, device)
    
    # Run evaluation
    evaluation_on_cadb(model, cfg)
