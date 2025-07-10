import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, trunc_normal_
import torch
class F2S(nn.Module):
    """ CNN feature maps -> Transformer patch embeddings
    """

    def __init__(self, inplanes, outplanes, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(F2S, self).__init__()
        self.dw_stride = 1
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=self.dw_stride, padding=0)
        # self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)
        self.ln = norm_layer(inplanes)
        self.act = act_layer()

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # [N, C, H, W]- [N, H*W, C]
        x = self.ln(x)
        x = self.act(x)
        return x

class S2F(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, inplanes, outplanes, stride, act_layer=nn.PReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), ):
        super(S2F, self).__init__()
        self.stride = stride
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=self.stride, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, H, W):
        B, _, C = x.shape
        x_r = x.transpose(1, 2).reshape(B, C, H, W)
        x_r = self.act(self.bn(self.conv_project(x_r)))
        return x_r
    

    


class Block_unit_MHSA_pro(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        #MHSA
        # self.attn = Attention(dim, num_heads=num_heads,  attn_drop=attn_drop, proj_drop=drop)
        ###MHSApro
        self.attn = Attention_pro(dim, num_heads=num_heads,  attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm = norm_layer(dim)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
    def forward(self, x):
        # x = x + self.drop_path(self.attn(self.norm(x)))
        x = self.norm(self.proj(x)) + self.drop_path(self.attn(x))
        
        # x = self.proj(x) + self.drop_path(self.attn(x))
        # x = self.drop_path(self.attn(self.norm(x)))
        return x
    

class Attention_pro(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  #(3,B,Hs,N,C/Hs)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q0_1, k0_1, v0_1 = torch.roll(q, shifts=-1,dims=2), torch.roll(k, shifts=-1,dims=2), torch.roll(v, shifts=-1,dims=2)
        q1_0, k1_0, v1_0 = torch.roll(q, shifts=1,dims=2), torch.roll(k, shifts=1,dims=2), torch.roll(v, shifts=1,dims=2)
        q = torch.mean(torch.stack([q,q0_1,q1_0]), dim=0)
        k = torch.mean(torch.stack([k,k0_1,k1_0]), dim=0)
        v = torch.mean(torch.stack([v,v0_1,v1_0]), dim=0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

    
class MHSAblock_pro(nn.Module):
    def __init__(self, inplanes, planes, stride=1, use_att=False,
                 embed_dim=256, num_heads=4, mlp_ratio=4.,qkv_bias=True,
                 qk_scale=None, drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0, fcu=False):
        super(MHSAblock_pro, self).__init__()
        self.squeeze_block = F2S(inplanes=inplanes, outplanes=planes)
        self.expand_block = S2F(inplanes=inplanes, outplanes=planes, stride=stride)
        self.trans_block = Block_unit_MHSA_pro(
            dim=inplanes, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)
        
    def forward(self, x):
        _, _, H, W = x.shape
        x = self.squeeze_block(x)
        x = self.trans_block(x)
        x = self.expand_block(x, H, W)
        return x
    
