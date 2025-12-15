import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import os
from commons import reflect_conv
from Networks.trm import Block, Decoder_Transformer
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0')
from Networks.module import FEM
def Fusion(vi_out, ir_out):
    return torch.cat([vi_out, ir_out], dim=1)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.vi_conv1 = nn.Conv2d(in_channels=1, kernel_size=1, out_channels=8, stride=1, padding=0)
        self.ir_conv1 = nn.Conv2d(in_channels=1, kernel_size=1, out_channels=8, stride=1, padding=0)

        self.vi_conv2 = reflect_conv(in_channels=8, kernel_size=3, out_channels=16, stride=1, pad=1)
        self.ir_conv2 = reflect_conv(in_channels=8, kernel_size=3, out_channels=16, stride=1, pad=1)

        self.vi_conv3 = reflect_conv(in_channels=16, kernel_size=3, out_channels=32, stride=1, pad=1)
        self.ir_conv3 = reflect_conv(in_channels=16, kernel_size=3, out_channels=32, stride=1, pad=1)

        self.vi_conv4 = reflect_conv(in_channels=32, kernel_size=3, out_channels=64, stride=1, pad=1)
        self.ir_conv4 = reflect_conv(in_channels=32, kernel_size=3, out_channels=64, stride=1, pad=1)

        self.trans1 = FEM(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.trans2 = FEM(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.trans3 = FEM(in_channels=64, out_channels=64, kernel_size=3, padding=1)


        self.conv1x1_32 = nn.Conv2d(32, 64, kernel_size=1)
        self.conv1x1_64 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv1x1_128 = nn.Conv2d(128, 64, kernel_size=1)
    def forward(self, vis_y_image, ir_image):
        activate = nn.GELU()
        activate = activate.to(device)
        vi_out = activate(self.vi_conv1(vis_y_image))
        ir_out = activate(self.ir_conv1(ir_image)) # [16, 8, 64, 64]
        vi_1 = self.vi_conv2(vi_out)
        ir_1 = self.ir_conv2(ir_out)
        vi_out1, ir_out1 = self.trans1(activate(vi_1), activate(ir_1))
        F1 = self.conv1x1_32(Fusion(vi_out1, ir_out1))

        vi_2 = self.vi_conv3(vi_out1)
        ir_2 = self.ir_conv3(ir_out1)
        vi_out2, ir_out2 = self.trans2(activate(vi_2), activate(ir_2))
        F2 = self.conv1x1_64(Fusion(vi_out2, ir_out2))

        vi_3 = self.vi_conv4(vi_out2)
        ir_3 = self.ir_conv4(ir_out2)
        vi_out3, ir_out3 = self.trans3(activate(vi_3), activate(ir_3))
        F3 = self.conv1x1_128(Fusion(vi_out3, ir_out3))

        return F1, F2, F3
    


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.conv1 = reflect_conv(in_channels=32, kernel_size=3, out_channels=16, stride=1, pad=1)  # 减半输出通道数
        self.conv2 = reflect_conv(in_channels=16, kernel_size=3, out_channels=16, stride=1, pad=1)  # 减半输出通道数
        self.conv3 = reflect_conv(in_channels=16, kernel_size=3, out_channels=8, stride=1, pad=1)  # 减半输出通道数
        self.transformer_block3 = Block(dim=8, num_heads=8, ffn_expansion_factor=4, bias=False,
                                                  LayerNorm_type='WithBias')  # 减半num_heads
        self.transformer_block3 = self.transformer_block3.cuda()
        self.conv4 = nn.Conv2d(in_channels=8, kernel_size=1, out_channels=1, stride=1, padding=0)

    def forward(self, x):
        activate = nn.GELU()
        activate = activate.cuda()
        x = activate(self.conv1(x))
        x = activate(self.conv2(x))
        x = activate(self.conv3(x))
        x = activate(self.transformer_block3(x))
        x = nn.Tanh()(self.conv4(x)) / 2 + 0.5
        return x
    
class GatingNetwork(nn.Module):
    def __init__(self, in_channels=128, num_experts=3):
        super(GatingNetwork, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, num_experts, kernel_size=1),
            nn.Softmax(dim=1)  # [B, 3, 1, 1]
        )

    def forward(self, x):
        weights = self.global_pool(x)  # [B, C, 1, 1]
        weights = self.fc(weights)     # [B, 3, 1, 1]
        return weights

class MODEL(nn.Module):
    def __init__(self):
        super(MODEL, self).__init__()

        self.low_encoder = Encoder()
        self.gating = GatingNetwork(in_channels=192, num_experts=3) 
        self.convm = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1)
        self.fusion = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)
        self.decoder_transformer = Decoder_Transformer()
        self.decoder = Decoder()

    def forward(self, vis_y_image, ir_image):
        # 获取三个专家输出
        F1, F2, F3 = self.low_encoder(vis_y_image, ir_image)  # [B, C, H, W] x 3
        experts = torch.cat([F1, F2, F3 ], dim=1)  # [B, 64*3=192, H, W]

        # 获取门控权重
        weights = self.gating(experts)  # [B, 3, 1, 1]

        # 分别加权每个专家
        F1_weighted = F1 * weights[:, 0:1, :, :]
        F2_weighted = F2 * weights[:, 1:2, :, :]
        F3_weighted = F3 * weights[:, 2:3, :, :]

        moe = torch.cat([F1_weighted, F2_weighted, F3_weighted], dim=1)  # [B, 192, H, W]

        # 后续处理
        low_freq_fused = self.convm(moe)                   # [B, 64, H, W]
        low_freq_fused1 = self.decoder_transformer(low_freq_fused)
        fused_output = self.fusion(low_freq_fused1)       # [B, 32, H, W]
        fused_image = self.decoder(fused_output)          # [B, 1, H, W]
        return fused_image