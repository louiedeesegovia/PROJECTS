import torch
from thop import profile
from config import Config
from samp_net import SAMPNet

cfg = Config()
names = ['resnet18', 'mobilenet_v3_small', 'mobilenet_v3_large',
         'mobilenet_v2', 'efficientnet_b0', 'shufflenet_v2_x1_0']
for name in names:
    for proj in ([0, 512] if name == 'resnet18' else [512]):
        cfg.backbone, cfg.proj_channels = name, proj
        m = SAMPNet(cfg, pretrained=False).eval()
        x = torch.randn(2, 3, 224, 224)
        s = torch.rand(2, 1, 224, 224)
        with torch.no_grad():
            fm = m.proj(m.backbone(x))
            _, _, out = m(x, s)
        nb = sum(p.numel() for p in m.backbone.parameters()) / 1e6
        nt = sum(p.numel() for p in m.parameters()) / 1e6
        macs, _ = profile(m.backbone, inputs=(torch.randn(1, 3, 224, 224),), verbose=False)
        print('%-20s proj=%-4d fmap=%s backbone=%5.2fM total=%5.2fM bbGMACs=%.2f out=%s'
              % (name, proj, tuple(fm.shape[1:]), nb, nt, macs / 1e9, tuple(out.shape)))
