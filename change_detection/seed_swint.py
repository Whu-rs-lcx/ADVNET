import timm
import torch
from einops import rearrange
from typing import List, Optional

from mmseg.registry import MODELS
import torch
import os
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS

from mmseg.models.backbones.resnet import BasicBlock
from timm.models.swin_transformer_v2 import PatchMerging, SwinTransformerV2Block
import torch.utils.checkpoint as checkpoint
from typing import Tuple, Union
from timm.layers import to_2tuple

_int_or_tuple_2_t = Union[int, Tuple[int, int]]

class SwinTV2Block(nn.Module):
    def __init__(
            self,
            dim: int,
            out_dim: int,
            input_resolution: _int_or_tuple_2_t,
            depth: int=2,
            num_heads: int=8,
            window_size: _int_or_tuple_2_t=8,
            downsample: bool = False,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            proj_drop: float = 0., 
            attn_drop: float = 0.,
            drop_path: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            pretrained_window_size: _int_or_tuple_2_t = 0,
            output_nchw: bool = False,
    ) -> None:
        """
        Args:
            dim: Number of input channels.
            out_dim: Number of output channels.
            input_resolution: Input resolution.
            depth: Number of blocks.
            num_heads: Number of attention heads.
            window_size: Local window size.
            downsample: Use downsample layer at start of the block.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            proj_drop: Projection dropout rate
            attn_drop: Attention dropout rate.
            drop_path: Stochastic depth rate.
            norm_layer: Normalization layer.
            pretrained_window_size: Local window size in pretraining.
            output_nchw: Output tensors on NCHW format instead of NHWC.
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.output_resolution = tuple(i // 2 for i in input_resolution) if downsample else input_resolution
        self.depth = depth
        self.output_nchw = output_nchw
        self.grad_checkpointing = False
        window_size = to_2tuple(window_size)
        shift_size = tuple([w // 2 for w in window_size])

        # patch merging / downsample layer
        if downsample:
            self.downsample = PatchMerging(dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            assert dim == out_dim
            self.downsample = nn.Identity()

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerV2Block(
                dim=out_dim,
                input_resolution=self.output_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size,
            )
            for i in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)

        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x.permute(0, 3, 1, 2)

    def _init_respostnorm(self) -> None:
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResDecodeBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.block = self._make_layer(BottleNeck, self.in_channels, 2, 1)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SEED_SwinT(nn.Module):
    def __init__(self, neck=None, feature_exchange_type='le'):
        super(SEED_SwinT, self).__init__()
        try:
            self.model = timm.create_model('swinv2_base_window8_256', pretrained=True, pretrained_cfg_overlay=dict(file=os.path.join(os.environ.get("PRETRAIN"), 'swinv2_base_window8_256.ms_in1k/pytorch_model.bin')), features_only=True)
        except:
            self.model = timm.create_model('swinv2_base_window8_256', pretrained=True, features_only=True)
        '''
        swinv2_base_window8_256
        torch.Size([16, 64, 64, 128])
        torch.Size([16, 32, 32, 256])
        torch.Size([16, 16, 16, 512])
        torch.Size([16, 8, 8, 1024])
        '''
        self.interaction_layers = ['patch_embed', 'layers_2', 'layers_3']

        FPN_DICT = {'type': 'FPN', 'in_channels': [128, 256, 512, 1024], 'out_channels': 256, 'num_outs': 4}
        self.fpnA = MODELS.build(FPN_DICT)
        # self.fpnB = MODELS.build(FPN_DICT)

        self.decode_layersA = nn.Sequential(
            nn.Identity(),
            SwinTV2Block(dim=FPN_DICT['out_channels'], out_dim=FPN_DICT['out_channels'], input_resolution=64, num_heads=4),
            SwinTV2Block(dim=FPN_DICT['out_channels'], out_dim=FPN_DICT['out_channels'], input_resolution=32, num_heads=8),
            SwinTV2Block(dim=FPN_DICT['out_channels'], out_dim=FPN_DICT['out_channels'], input_resolution=16, num_heads=16),
            SwinTV2Block(dim=FPN_DICT['out_channels'], out_dim=FPN_DICT['out_channels'], input_resolution=8, num_heads=32)
        )
        self.feature_exchange_type = feature_exchange_type
        self.decode_conv = nn.Sequential(
            nn.Conv2d(FPN_DICT['out_channels'], FPN_DICT['out_channels'], kernel_size=3, padding=1),
            nn.BatchNorm2d(FPN_DICT['out_channels']),
            nn.ReLU(inplace=True)
        )
        self.conv_seg = nn.Conv2d(FPN_DICT['out_channels'], 2, kernel_size=1)

    def layer_exchange(self, x, y):
        i = 2
        for index in range(0, len(x), i):
            x[index], y[index] = y[index], x[index]
        return x, y

    def random_layer_exchange(self, x, y, exchange_threshold=0.5):
        for i in range(len(x)):
            # 对每一层生成一个随机数
            if torch.rand(1).item() < exchange_threshold:
                x[i], y[i] = y[i], x[i]
        return x, y

    def channel_exchange(self, x1, x2, p=2):
        N, C, H, W = x1.shape
        exchange_mask = torch.arange(C) % p == 0
        exchange_mask1 = exchange_mask.cuda().int().expand((N, C)).unsqueeze(-1).unsqueeze(-1)  # b,c,1,1
        exchange_mask2 = 1 - exchange_mask1
        out_x1 = exchange_mask1 * x1 + exchange_mask2 * x2
        out_x2 = exchange_mask1 * x2 + exchange_mask2 * x1

        return out_x1, out_x2

    def random_channel_exchange(self, x1, x2, exchange_ratio=0.5):
        N, C, H, W = x1.shape
        
        # 随机生成置换通道的索引
        perm = torch.randperm(C)  # 生成随机排列的通道索引
        num_exchange = int(C * exchange_ratio)  # 计算需要置换的通道数量
        exchange_indices = perm[:num_exchange]  # 随机选择部分通道的索引
        
        # 构造通道掩码
        exchange_mask = torch.zeros(C, dtype=torch.bool, device=x1.device)
        exchange_mask[exchange_indices] = True  # 标记需要置换的通道
        exchange_mask1 = exchange_mask.view(1, C, 1, 1)  # 扩展为 (1, C, 1, 1)
        exchange_mask2 = ~exchange_mask1  # 补集，表示不置换的通道

        # 执行通道置换
        out_x1 = exchange_mask1 * x2 + exchange_mask2 * x1
        out_x2 = exchange_mask1 * x1 + exchange_mask2 * x2

        return out_x1, out_x2

    def spatial_exchange(self, x1, x2, p=2):
        N, c, h, w = x1.shape
        # 构造掩码，并扩展维度以便在所有 batch、channel 和 height 上广播
        mask = (torch.arange(w, device=x1.device) % p == 0).view(1, 1, 1, w)
        out_x1 = torch.where(mask, x2, x1)
        out_x2 = torch.where(mask, x1, x2)
        return out_x1, out_x2

    def random_spatial_exchange(self, x1, x2, p=2):
        N, c, h, w = x1.shape
        # 随机生成每个列是否交换的掩码，每个列被交换的概率为 1/p
        mask = (torch.rand(w, device=x1.device) < (1.0 / p)).view(1, 1, 1, w)
        out_x1 = torch.where(mask, x2, x1)
        out_x2 = torch.where(mask, x1, x2)
        return out_x1, out_x2

    def feature_exchange(self, xA_list, xB_list, feature_exchange_type='le'):
        if feature_exchange_type=='le':
            xA_list, xB_list = self.layer_exchange(xA_list, xB_list)
        elif feature_exchange_type=='rle':
            if self.training:
                xA_list, xB_list = self.random_layer_exchange(xA_list, xB_list)
            else:
                xA_list, xB_list = self.layer_exchange(xA_list, xB_list)
        elif feature_exchange_type=='ce':
            xA_list[2], xB_list[2] = self.channel_exchange(xA_list[2], xB_list[2], p=2)
            xA_list[3], xB_list[3] = self.channel_exchange(xA_list[3], xB_list[3], p=2)
        elif feature_exchange_type=='se':
            xA_list[2], xB_list[2] = self.spatial_exchange(xA_list[2], xB_list[2], p=2)
            xA_list[3], xB_list[3] = self.spatial_exchange(xA_list[3], xB_list[3], p=2)
        elif feature_exchange_type=='rse':
            if self.training:
                xA_list[2], xB_list[2] = self.random_spatial_exchange(xA_list[2], xB_list[2], p=2)
                xA_list[3], xB_list[3] = self.random_spatial_exchange(xA_list[3], xB_list[3], p=2)
            else:
                xA_list[2], xB_list[2] = self.spatial_exchange(xA_list[2], xB_list[2], p=2)
                xA_list[3], xB_list[3] = self.spatial_exchange(xA_list[3], xB_list[3], p=2)
        elif feature_exchange_type=='rce':
            if self.training:
                xA_list[2], xB_list[2] = self.random_channel_exchange(xA_list[2], xB_list[2], exchange_ratio=0.5)
                xA_list[3], xB_list[3] = self.random_channel_exchange(xA_list[3], xB_list[3], exchange_ratio=0.5)
            else:
                xA_list[2], xB_list[2] = self.channel_exchange(xA_list[2], xB_list[2], p=2)
                xA_list[3], xB_list[3] = self.channel_exchange(xA_list[3], xB_list[3], p=2)
        else:
            if self.training:
                xA_list, xB_list = self.random_layer_exchange(xA_list, xB_list)
            else:
                xA_list, xB_list = self.layer_exchange(xA_list, xB_list)
        return xA_list, xB_list

    def decode_stage(self, feature_list):
        x1, x2, x3, x4 = feature_list
        x4 = self.decode_layersA[4](x4)
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x3 = x3 + x4.permute(0, 2, 3, 1)
        x3 = self.decode_layersA[3](x3)
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x2 = x2 + x3.permute(0, 2, 3, 1)
        x2 = self.decode_layersA[2](x2)
        x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
        x1 = x1 + x2.permute(0, 2, 3, 1)
        x1 = self.decode_layersA[1](x1)
        return x1

    def decode_head(self, x):
        x = self.decode_conv(x)
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        x = self.conv_seg(x)
        return x

    def forward(self, xA, xB):
        xA_list = self.model(xA)
        xB_list = self.model(xB)
        xA_list = [x.permute(0, 3, 1, 2) for x in xA_list]
        xB_list = [x.permute(0, 3, 1, 2) for x in xB_list]
        
        xA_list, xB_list = self.feature_exchange(xA_list, xB_list, feature_exchange_type=self.feature_exchange_type)

        xA_list = self.fpnA(xA_list)
        xB_list = self.fpnA(xB_list)

        xA_list = [x.permute(0, 2, 3, 1) for x in xA_list]
        xB_list = [x.permute(0, 2, 3, 1) for x in xB_list]
        outA = self.decode_stage(xA_list)
        outB = self.decode_stage(xB_list)

        outA = self.decode_head(outA)
        outB = self.decode_head(outB)
        return outA, outB



if __name__ == '__main__':
    # Example usage
    model = SEED_SwinT(feature_exchange_type='le')
    xA = torch.randn(1, 3, 256, 256)
    xB = torch.randn(1, 3, 256, 256)
    output = model(xA, xB)
    print([o.shape for o in output])