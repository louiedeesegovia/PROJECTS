import torch
from config import Config
from samp_net import SAMPNet

cfg = Config()
for name in ['resnet18', 'mobilenet_v3_small', 'mobilenet_v3_large',
             'shufflenet_v2_x1_0', 'mobilenet_v2', 'efficientnet_b0']:
    cfg.backbone, cfg.proj_channels = name, 0
    m = SAMPNet(cfg, pretrained=False).eval()
    nb = sum(p.numel() for p in m.backbone.parameters()) / 1e6
    nt = sum(p.numel() for p in m.parameters()) / 1e6
    with torch.no_grad():
        c = m.backbone(torch.randn(1, 3, 224, 224)).shape[1]
    print('%-20s C=%-5d backbone=%6.2fM total=%7.2fM head=%7.2fM' % (name, c, nb, nt, nt - nb))
