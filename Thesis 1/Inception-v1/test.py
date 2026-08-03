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

def benchmark_model(model, device):
    """
    Benchmark the model for FLOPs, parameters, memory usage, and FPS
    """
    print("\n" + "="*60)
    print("MODEL BENCHMARKING")
    print("="*60)
    
    model.eval()
    dummy_image = torch.randn(1, 3, 224, 224).to(device)
    dummy_saliency = torch.randn(1, 1, 224, 224).to(device)
    
    # 1. Calculate Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n1. MODEL PARAMETERS:")
    print(f"   Total Parameters: {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    print(f"   Non-trainable Parameters: {total_params - trainable_params:,}")
    
    # 2. Calculate FLOPs using thop
    try:
        from thop import profile, clever_format
        flops, params = profile(model, inputs=(dummy_image, dummy_saliency), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        print(f"\n2. FLOPs (Floating Point Operations):")
        print(f"   FLOPs: {flops}")
        print(f"   Params (from thop): {params}")
    except ImportError:
        print("\n2. FLOPs: thop library not installed. Install with: pip install thop")
    except Exception as e:
        print(f"\n2. FLOPs: Error calculating FLOPs: {e}")
    
    # 3. GPU Memory Usage
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        with torch.no_grad():
            _ = model(dummy_image, dummy_saliency)
        torch.cuda.synchronize()
        memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
        print(f"\n3. GPU MEMORY USAGE:")
        print(f"   Peak Memory Allocated: {memory_allocated:.2f} MB")
        print(f"   Peak Memory Reserved: {memory_reserved:.2f} MB")
    else:
        print("\n3. GPU MEMORY USAGE: Not available (running on CPU)")
    
    # 4. Inference Speed (FPS)
    print(f"\n4. INFERENCE SPEED:")
    with torch.no_grad():
        # Warmup
        print("   Warming up...")
        for _ in range(20):
            _ = model(dummy_image, dummy_saliency)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Actual timing
        print("   Measuring inference speed...")
        num_iterations = 100
        start_time = time.time()
        for _ in range(num_iterations):
            _ = model(dummy_image, dummy_saliency)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        total_time = end_time - start_time
        fps = num_iterations / total_time
        avg_latency = (total_time / num_iterations) * 1000  # ms
        
        print(f"   Iterations: {num_iterations}")
        print(f"   Total Time: {total_time:.4f} seconds")
        print(f"   Average Latency: {avg_latency:.2f} ms/image")
        print(f"   Throughput (FPS): {fps:.2f} images/second")
    
    print("\n" + "="*60)
    print()
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'fps': fps if device.type == 'cuda' else None,
        'memory_mb': memory_allocated if device.type == 'cuda' else None
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
    model = SAMPNet(cfg, pretrained=False).to(device)
    
    # Run benchmarks
    benchmark_results = benchmark_model(model, device)
    
    # If you have a pretrained weight file, uncomment these lines:
    weight_file = 'C:/Users/segov/Music/SAMP-Net/SAMPNet/experiments/inception_v1_samp_aaff_wemd/checkpoints/model-best.pth'
    model.load_state_dict(torch.load(weight_file))
    evaluation_on_cadb(model, cfg)
