"""Explicit MobileNetV2 implementation and controlled convolution/SE ablations.

Architecture: Sandler et al., CVPR 2018. The unmodified model's state-dict keys
are compatible with torchvision.models.mobilenet_v2 (including its classifier).
No pretrained weights are used by this coursework experiment.
"""
import hashlib
import math
import torch
from torch import nn

VARIANTS = ('mobilenet_v2', 'mobilenet_v2_dense', 'mobilenet_v2_se', 'mobilenet_v2_dense_se',
            'mobilenet_v2_gelu', 'mobilenet_v2_mish', 'mobilenet_v2_gn', 'mobilenet_v2_cbam')

def normalization(channels, kind):
    return nn.GroupNorm(math.gcd(32, channels), channels) if kind == 'gn' else nn.BatchNorm2d(channels)

def activation(kind):
    if kind == 'gelu': return nn.GELU(approximate='none')
    if kind == 'mish': return nn.Mish()
    return nn.ReLU6(inplace=True)

class ConvBNReLU6(nn.Sequential):
    def __init__(self, inp, out, kernel=3, stride=1, groups=1, act='relu6', norm='bn'):
        super().__init__(nn.Conv2d(inp, out, kernel, stride, (kernel-1)//2, groups=groups, bias=False),
                         normalization(out, norm), activation(act))

class SE(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(channels, max(1, channels // reduction), 1)
        self.relu = nn.ReLU(inplace=True)
        self.expand = nn.Conv2d(max(1, channels // reduction), channels, 1)
    def forward(self, x):
        return x * torch.sigmoid(self.expand(self.relu(self.reduce(self.pool(x)))))

class CBAM(nn.Module):
    """Sequential channel (shared average/max MLP) and 7x7 spatial attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Conv2d(channels, max(1, channels//reduction), 1),
                                 nn.ReLU(), nn.Conv2d(max(1, channels//reduction), channels, 1))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)
    def forward(self, x):
        weights = torch.sigmoid(self.mlp(x.mean((2,3), keepdim=True)) + self.mlp(x.amax((2,3), keepdim=True)))
        x = x * weights
        spatial = torch.cat([x.mean(1,keepdim=True), x.amax(1,keepdim=True)], dim=1)
        return x * torch.sigmoid(self.spatial(spatial))

class InvertedResidual(nn.Module):
    def __init__(self, inp, out, stride, expansion, dense=False, attention=False, act='relu6', norm='bn'):
        super().__init__()
        hidden = int(round(inp * expansion))
        self.use_res_connect = stride == 1 and inp == out
        layers = []
        if expansion != 1: layers.append(ConvBNReLU6(inp, hidden, kernel=1, act=act, norm=norm))
        layers.extend([ConvBNReLU6(hidden, hidden, stride=stride, groups=1 if dense else hidden, act=act, norm=norm),
                       nn.Conv2d(hidden, out, 1, bias=False), normalization(out, norm)])
        self.conv = nn.Sequential(*layers)
        self.se = CBAM(out) if attention == 'cbam' else SE(out) if attention else nn.Identity()
    def forward(self, x):
        y = self.se(self.conv(x))
        return x + y if self.use_res_connect else y

class MobileNetV2(nn.Module):
    def __init__(self, variant='mobilenet_v2', num_classes=6):
        super().__init__()
        if variant not in VARIANTS: raise ValueError(variant)
        dense, attention = 'dense' in variant, 'cbam' if variant.endswith('_cbam') else variant.endswith('_se')
        act = 'gelu' if variant.endswith('_gelu') else 'mish' if variant.endswith('_mish') else 'relu6'
        norm = 'gn' if variant.endswith('_gn') else 'bn'
        setting = [(1,16,1,1), (6,24,2,2), (6,32,3,2), (6,64,4,2),
                   (6,96,3,1), (6,160,3,2), (6,320,1,1)]
        inp = 32; features = [ConvBNReLU6(3, inp, stride=2, act=act, norm=norm)]
        for expansion, out, repeats, stride in setting:
            for i in range(repeats):
                features.append(InvertedResidual(inp, out, stride if i == 0 else 1, expansion, dense, attention, act, norm))
                inp = out
        features.append(ConvBNReLU6(inp, 1280, kernel=1, act=act, norm=norm))
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, num_classes))
        self.variant = variant
    def forward(self, x):
        return self.classifier(self.features(x).mean((2,3)))

def initialize(model, seed):
    """Identical shared modules receive identical initialization across variants.

    Each module has an independent RNG seed derived from its name. Adding SE
    therefore does not shift random initialization of the following layers.
    """
    for name, module in model.named_modules():
        module_seed = int.from_bytes(hashlib.sha256(f'{seed}:{name}'.encode()).digest()[:8], 'big') % (2**31)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(module_seed)
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if module.bias is not None: nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight); nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01); nn.init.zeros_(module.bias)
    return model

def build_model(variant, seed=42):
    return initialize(MobileNetV2(variant), seed)
