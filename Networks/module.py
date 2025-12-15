from torch import nn
import torch
import torch.nn.functional as F
from typing import Optional, Callable
from functools import partial
from Networks.trm import BasicLayer, PatchEmbed
from scipy.fftpack import dct
import numpy as np
class Enhancement_texture_LDC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        # conv.weight.size() = [out_channels, in_channels, kernel_size, kernel_size]
        super(Enhancement_texture_LDC, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)  # [12,3,3,3]

        self.register_buffer('center_mask', torch.tensor([[0, 0, 0],
                                                    [0, 1, 0],
                                                    [0, 0, 0]], dtype=torch.float))
        self.base_mask = nn.Parameter(torch.ones(self.conv.weight.size()), requires_grad=False)  # [12,3,3,3]
        self.learnable_mask = nn.Parameter(torch.ones([self.conv.weight.size(0), self.conv.weight.size(1)]),
                                           requires_grad=True)  # [12,3]
        self.learnable_theta = nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)  # [1]

    def forward(self, x):
        mask = self.base_mask - self.learnable_theta * self.learnable_mask[:, :, None, None] * \
               self.center_mask * self.conv.weight.sum(2).sum(2)[:, :, None, None]

        out_diff = F.conv2d(input=x, weight=self.conv.weight * mask, bias=self.conv.bias, stride=self.conv.stride,
                            padding=self.conv.padding,
                            groups=self.conv.groups)
        return out_diff
class Differential_enhance(nn.Module):
    def __init__(self, nf=48):
        super(Differential_enhance, self).__init__()

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.act = nn.Sigmoid()
        self.lastconv = nn.Conv2d(nf,nf//2,1,1)

    def forward(self, fuse, x1, x2):
        b,c,h,w = x1.shape
        sub_1_2 = x1 - x2
        sub_w_1_2 = self.global_avgpool(sub_1_2)
        w_1_2 = self.act(sub_w_1_2)
        sub_2_1 = x2 - x1
        sub_w_2_1 = self.global_avgpool(sub_2_1)
        w_2_1 = self.act(sub_w_2_1)
        D_F1 = torch.multiply(w_1_2, fuse)
        D_F2 = torch.multiply(w_2_1, fuse)
        F_1 = torch.add(D_F1, other=x1, alpha=1)
        F_2 = torch.add(D_F2, other=x2, alpha=1)

        return F_1, F_2
class Cross_layer(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0
    ):
        super().__init__()
        self.d_model = hidden_dim
        self.texture_enhance1 = Enhancement_texture_LDC(self.d_model, self.d_model)
        self.texture_enhance2 = Enhancement_texture_LDC(self.d_model, self.d_model)
        self.Diff_enhance = Differential_enhance(self.d_model)

    def forward(self, Fuse, x1,x2):
        TX_x1 = self.texture_enhance1(x1)
        TX_x2 = self.texture_enhance2(x2)

        DF_x1, DF_x2 = self.Diff_enhance(Fuse, x1,x2)
        F_1 = TX_x1 +DF_x1
        F_2 = TX_x2 +DF_x2
        F = F_1 + F_2
        return F
    


def dct_energy(x, top_k=16):
    """对整个图像做 DCT 并返回 top-k 能量"""
    B, C, H, W = x.shape
    x_np = x.detach().cpu().numpy().reshape(B * C, H, W)
    dct_out = []

    for i in range(B * C):
        dct_2d = dct(dct(x_np[i], axis=0, norm='ortho'), axis=1, norm='ortho')
        dct_out.append(dct_2d)

    dct_out = np.stack(dct_out).reshape(B, C, H, W)
    dct_energy = np.abs(dct_out).mean(axis=1)  # (B, H, W)

    # 展平选 top-k
    dct_energy_flat = dct_energy.reshape(B, -1)
    topk_vals = np.partition(dct_energy_flat, -top_k, axis=-1)[:, -top_k:]
    return torch.tensor(topk_vals, dtype=x.dtype, device=x.device)  # (B, top_k)

class ECA(nn.Module):
    def __init__(self, top_k=16, reduction=4):
        super(ECA, self).__init__()
        self.top_k = top_k
        self.fc = nn.Sequential(
            nn.Linear(top_k, top_k // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(top_k // reduction, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 频域能量 top-k
        freq_feat = dct_energy(x, self.top_k)  # (B, top_k)
        att_weight = self.fc(freq_feat)  # (B, 1)

        # 扩展为 (B, 1, 1, 1) 后与输入相乘
        return x * att_weight.view(-1, 1, 1, 1)
class FEM(nn.Module):
    def __init__(
            self,
            in_channels: int,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(in_channels)  
        self.ln_2 = norm_layer(in_channels)  
        self.Cross_layer = Cross_layer(in_channels)
        self.self_attention_cross_spatial = ECA(in_channels)
        self.patch_embed = PatchEmbed(
            img_size=256, patch_size=4, in_chans=in_channels, embed_dim=96,
            norm_layer=norm_layer)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.basicLayer=BasicLayer(dim= in_channels,
                                   input_resolution=(patches_resolution[0],patches_resolution[1]),
                                         depth=1,
                                         num_heads=8,
                                         window_size=1,
                                         mlp_ratio=4,
                                         qkv_bias=True, qk_scale=None,
                                         drop=0, attn_drop=0,
                                         drop_path=0,
                                         norm_layer=nn.LayerNorm,
                                         downsample=None,
                                         use_checkpoint=False)
    def forward(self, input1: torch.Tensor, input2:torch.Tensor):
        x_1 = input1  
        x_2 = input2  

        Fuse = torch.add(x_1, x_2, alpha=1)
        F = self.Cross_layer(Fuse, x_1, x_2)
        F.size = (F.shape[2], F.shape[3])
        Cross_x1x2 = self.basicLayer(F, F.size) #(b, h, w, c)
        Cross_x1x2_ = Cross_x1x2.permute(0, 3, 1, 2) #(b, c, h, w )
        Cross_x1x2_spatial =  self.self_attention_cross_spatial(Cross_x1x2_) #(b, c, h, w)
        Cross_x1x2_spatial = Cross_x1x2_spatial.permute(0, 2, 3, 1)


        x_1 = x_1 + Cross_x1x2 + Cross_x1x2_spatial
        x_2 = x_2 + Cross_x1x2 + Cross_x1x2_spatial
        return x_1, x_2
