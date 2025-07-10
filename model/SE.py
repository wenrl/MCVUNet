import torch
import torch.nn as nn
import torch.nn.functional as F

class LiteSqueezeExcitation(nn.Module):

    def __init__(self, channels, reduction=4, groups=4):
        super().__init__()
        self.groups = groups
        reduced_channels = max(4, channels // reduction)  # 确保最小通道数
        
        # 轻量化Squeeze (深度可分离+分组)
        self.squeeze = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=False),  # 分组Pointwise
            nn.BatchNorm2d(channels)
        )
        
        # 零参数字符激励 (基于移位操作)
        self.excite = nn.Conv2d(channels, channels, 1,  bias=True)
        
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 分组压缩 (GAP + 分组1x1)
        y = self.squeeze(x)
        
        # 移位激励 (无需额外参数)
        y = y - y.mean(dim=1, keepdim=True)  # 中心化
        y = torch.sigmoid(self.excite(y))    # 分组激励
        
        # 动态门控 (学习全局重要性)
        return x * y + x # [B,C,1,1] * [B,C,1,1] * [1,C,1,1]


import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):

    def __init__(self, channels, reduction=16, kernels=(3,5,7)):
        super().__init__()
        # Reduce to 1 channel
        self.reduce = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        # Depthwise separable convs with different kernel sizes
        self.dw_branches = nn.ModuleList()
        for k in kernels:
            pad = k // 2
            self.dw_branches.append(
                nn.Sequential(
                    nn.Conv2d(1, 1, kernel_size=k, padding=pad, groups=1, bias=False),
                    nn.Conv2d(1, 1, kernel_size=1, bias=False),
                    
                )
            )
        # Activation and sigmoid
        self.act = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B,C,H,W]
        B, C, H, W = x.size()
        # 1. Channel compression
        t = self.reduce(x)        # [B,1,H,W]
        # 2. Multi-scale depthwise conv
        outs = []
        for branch in self.dw_branches:
            o = branch(t)          # [B,1,H,W]
            outs.append(o)
        # 3. Fuse scales
        fused = sum(outs) / len(outs)  # [B,1,H,W]
        fused = self.act(fused)
        # 4. Produce attention map
        weight = self.sigmoid(fused)   # [B,1,H,W]
        # 5. Reweight original x
        return x * weight + x # broadcast on channel dim

# --------------- 轻量化对比测试 ------------------
if __name__ == '__main__':
    # 原始SE块
    original_se = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(256, 16, 1),
        nn.ReLU(),
        nn.Conv2d(16, 256, 1),
        nn.Sigmoid()
    )
    
    # 轻量化SE块
    lite_se = LiteSqueezeExcitation(256)
    
    # 计算量对比 (FLOPs)
    x = torch.randn(1, 256, 56, 56)
    print(f"Original SE FLOPs: {sum(m.flops for m in original_se.modules()) / 1e6:.2f}M")
    print(f"Lite SE FLOPs: {sum(m.flops for m in lite_se.modules()) / 1e6:.2f}M")
    
    # 参数量对比
    print(f"Original Params: {sum(p.numel() for p in original_se.parameters()) / 1e3:.1f}K")
    print(f"Lite Params: {sum(p.numel() for p in lite_se.parameters()) / 1e3:.1f}K")