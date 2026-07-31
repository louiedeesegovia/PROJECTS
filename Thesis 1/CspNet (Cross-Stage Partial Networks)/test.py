from samp_net import EMDLoss, SAMPNet
from cadb_dataset import CADBDataset
import torch
from torch.utils.data import DataLoader
import scipy.stats as stats
import numpy as np
from tqdm import tqdm
from config import Config
import time
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: thop not installed. Install with: pip install thop")

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
    # sort_target = np.sort(target_list).tolist()
    # sort_predict = np.sort(predict_list).tolist()
    # pre_rank = []
    # for i in predict_list:
    #     pre_rank.append(sort_predict.index(i))
    # tar_rank = []
    # for i in target_list:
    #     tar_rank.append(sort_target.index(i))
    # rho,pval = stats.spearmanr(pre_rank, tar_rank)
    rho,_ = stats.spearmanr(predict_list, target_list)
    return rho

def dist2ave(pred_dist):
    pred_score = torch.sum(pred_dist* torch.Tensor(range(1,6)).to(pred_dist.device), dim=-1, keepdim=True)
    return pred_score

def benchmark_model(model, cfg, device):
    """
    Benchmark model performance: FPS, FLOPs, Parameters, Memory Usage
    """
    print("\n" + "="*60)
    print("MODEL BENCHMARKING")
    print("="*60)
    
    model.eval()
    dummy_image = torch.randn(1, 3, 224, 224).to(device)
    dummy_saliency = torch.randn(1, 1, 224, 224).to(device)
    
    # 1. Measure Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n1. Model Parameters:")
    print(f"   Total Parameters: {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    print(f"   Model Size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # 2. Measure FLOPs
    if THOP_AVAILABLE:
        try:
            # Note: thop profile needs inputs as tuple
            flops, params = profile(model, inputs=(dummy_image, dummy_saliency), verbose=False)
            print(f"\n2. Computational Complexity:")
            print(f"   FLOPs: {flops:,} ({flops/1e9:.2f} GFLOPs)")
            print(f"   Params (from thop): {params:,}")
        except Exception as e:
            print(f"\n2. Computational Complexity:")
            print(f"   Error measuring FLOPs with thop: {e}")
            print(f"   Skipping FLOPs measurement")
    else:
        print(f"\n2. Computational Complexity:")
        print(f"   FLOPs: Not available (install thop: pip install thop)")
    
    # 3. Measure Inference Speed (FPS)
    print(f"\n3. Inference Speed:")
    with torch.no_grad():
        # Warmup
        for _ in range(20):
            _ = model(dummy_image, dummy_saliency)
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            _ = model(dummy_image, dummy_saliency)
        torch.cuda.synchronize()
        end = time.time()
        
        fps = 100 / (end - start)
        avg_time = (end - start) / 100 * 1000  # ms
        print(f"   Inference Time: {avg_time:.2f} ms per image")
        print(f"   Inference FPS: {fps:.2f} images/sec")
        print(f"   Throughput (batch={cfg.batch_size}): {fps * cfg.batch_size:.2f} images/sec")
    
    # 4. Measure GPU Memory Usage
    print(f"\n4. GPU Memory Usage:")
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(dummy_image, dummy_saliency)
    memory_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
    memory_reserved = torch.cuda.memory_reserved(device) / 1024**2
    print(f"   Peak Memory Allocated: {memory_allocated:.2f} MB")
    print(f"   Memory Reserved: {memory_reserved:.2f} MB")
    
    # 5. Batch Processing Benchmark
    print(f"\n5. Batch Processing (batch_size={cfg.batch_size}):")
    batch_image = torch.randn(cfg.batch_size, 3, 224, 224).to(device)
    batch_saliency = torch.randn(cfg.batch_size, 1, 224, 224).to(device)
    
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(batch_image, batch_saliency)
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(50):
            _ = model(batch_image, batch_saliency)
        torch.cuda.synchronize()
        end = time.time()
        
        batch_fps = (50 * cfg.batch_size) / (end - start)
        batch_time = (end - start) / 50 * 1000  # ms
        batch_memory = torch.cuda.max_memory_allocated(device) / 1024**2
        
        print(f"   Batch Time: {batch_time:.2f} ms per batch")
        print(f"   Batch Throughput: {batch_fps:.2f} images/sec")
        print(f"   Batch Memory: {batch_memory:.2f} MB")
    
    print("="*60 + "\n")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'fps': fps,
        'inference_time_ms': avg_time,
        'memory_mb': memory_allocated,
        'batch_throughput': batch_fps,
        'batch_memory_mb': batch_memory
    }

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
    model = SAMPNet(cfg,pretrained=False).to(device)
    weight_file = 'C:/Users/segov/Music/SAMP-Net/SAMPNet/experiments/cspnet_cspdarknet53_samp_aaff_wemd3/checkpoints/model-best.pth'
    model.load_state_dict(torch.load(weight_file))
    
    # Run benchmarking
    benchmark_stats = benchmark_model(model, cfg, device)
    
    # Run evaluation
    evaluation_on_cadb(model, cfg)