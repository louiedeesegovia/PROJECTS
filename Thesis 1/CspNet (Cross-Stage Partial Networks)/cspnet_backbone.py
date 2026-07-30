import torch
import torch.nn as nn
import torch.nn.functional as F

class Mish(nn.Module):
    """Mish activation function"""
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class ConvBNMish(nn.Module):
    """Convolution + BatchNorm + Mish activation"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBNMish, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.mish = Mish()
    
    def forward(self, x):
        return self.mish(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    """Residual block for CSPNet"""
    def __init__(self, channels, hidden_channels=None):
        super(ResidualBlock, self).__init__()
        if hidden_channels is None:
            hidden_channels = channels
        self.conv1 = ConvBNMish(channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBNMish(hidden_channels, channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return out

class CSPBlock(nn.Module):
    """Cross Stage Partial Block"""
    def __init__(self, in_channels, out_channels, num_blocks, first_block=False):
        super(CSPBlock, self).__init__()
        self.first_block = first_block
        
        if first_block:
            self.downsample = ConvBNMish(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.downsample = ConvBNMish(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        
        # Split channels for CSP
        mid_channels = out_channels // 2
        
        # Part 1: goes through residual blocks
        self.part1_conv = ConvBNMish(out_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(mid_channels, mid_channels // 2) for _ in range(num_blocks)
        ])
        self.blocks_conv = ConvBNMish(mid_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        
        # Part 2: shortcut
        self.part2_conv = ConvBNMish(out_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        
        # Transition
        self.transition = ConvBNMish(mid_channels * 2, out_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        x = self.downsample(x)
        
        # Split into two parts for CSP
        part1 = self.part1_conv(x)
        part2 = self.part2_conv(x)
        
        # Process part1 through residual blocks
        for block in self.blocks:
            part1 = block(part1)
        part1 = self.blocks_conv(part1)
        
        # Concatenate and transition
        out = torch.cat([part1, part2], dim=1)
        out = self.transition(out)
        
        return out

class CSPDarkNet53(nn.Module):
    """CSPDarkNet53 backbone"""
    def __init__(self, pretrained=False):
        super(CSPDarkNet53, self).__init__()
        
        # Initial convolution
        self.conv1 = ConvBNMish(3, 32, kernel_size=3, stride=1, padding=1)
        
        # CSP stages
        self.stage1 = CSPBlock(32, 64, num_blocks=1, first_block=True)
        self.stage2 = CSPBlock(64, 128, num_blocks=2)
        self.stage3 = CSPBlock(128, 256, num_blocks=8)
        self.stage4 = CSPBlock(256, 512, num_blocks=8)
        self.stage5 = CSPBlock(512, 1024, num_blocks=4)
        
        self._initialize_weights()
        
        if pretrained:
            print("Warning: Pretrained weights for CSPDarkNet53 not automatically loaded.")
            print("Please load pretrained weights manually if available.")
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        return x

class CSPResNet50(nn.Module):
    """CSPResNet50 backbone - lighter version"""
    def __init__(self, pretrained=False):
        super(CSPResNet50, self).__init__()
        
        # Initial convolution
        self.conv1 = ConvBNMish(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # CSP stages
        self.stage1 = CSPBlock(64, 256, num_blocks=3, first_block=False)
        self.stage2 = CSPBlock(256, 512, num_blocks=4)
        self.stage3 = CSPBlock(512, 1024, num_blocks=6)
        self.stage4 = CSPBlock(1024, 2048, num_blocks=3)
        
        self._initialize_weights()
        
        if pretrained:
            print("Warning: Pretrained weights for CSPResNet50 not automatically loaded.")
            print("Please load pretrained weights manually if available.")
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x

def build_cspnet(variant='cspdarknet53', pretrained=False):
    """
    Build CSPNet backbone
    Args:
        variant: 'cspdarknet53' or 'cspresnet50'
        pretrained: whether to load pretrained weights
    Returns:
        model: CSPNet backbone model
        out_channels: output channel dimension
    """
    if variant == 'cspdarknet53':
        model = CSPDarkNet53(pretrained=pretrained)
        out_channels = 1024
    elif variant == 'cspresnet50':
        model = CSPResNet50(pretrained=pretrained)
        out_channels = 2048
    else:
        raise ValueError(f"Unknown CSPNet variant: {variant}. Choose 'cspdarknet53' or 'cspresnet50'")
    
    return model, out_channels

if __name__ == '__main__':
    # Test the CSPNet backbone
    x = torch.randn(2, 3, 224, 224)
    
    print("Testing CSPDarkNet53...")
    model_dark, out_ch = build_cspnet('cspdarknet53')
    out = model_dark(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output channels: {out_ch}")
    
    print("\nTesting CSPResNet50...")
    model_res, out_ch = build_cspnet('cspresnet50')
    out = model_res(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output channels: {out_ch}")
