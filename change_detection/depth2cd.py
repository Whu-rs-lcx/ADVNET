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
import torch.utils.checkpoint as checkpoint
from timm.layers import to_2tuple
from depth_anything_v2.dpt import DepthAnythingV2

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


class Depth2CD(nn.Module):
    def __init__(self, neck=None, feature_exchange_type='le'):
        super(Depth2CD, self).__init__()

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

        self.model = DepthAnythingV2(**model_configs[encoder])
        # 1. 加载 checkpoint 的参数
        ckpt = torch.load(os.path.join(os.getenv('PRETRAIN'), f'checkpoints/depth_anything_v2_{encoder}.pth'), map_location='cpu')
        model_state = self.model.state_dict()

        # 2. 过滤掉 shape 不一致的参数（关键！）
        filtered_ckpt = {}
        for k, v in ckpt.items():
            if k in model_state and v.shape == model_state[k].shape:
                filtered_ckpt[k] = v  # 只保留shape对得上的参数
            else:
                print(f"Skip loading param, ckpt shape: {v.shape}")

        # 3. 加载过滤后的参数
        missing_keys, unexpected_keys = self.model.load_state_dict(filtered_ckpt, strict=False)
        print('Missing keys:', missing_keys)
        print('Unexpected keys:', unexpected_keys)
        self.sigmoid = nn.Sigmoid()

    def forward(self, xA, xB):

        depth, depthB = self.model.depth2cd(xA, xB)

        return depth, depthB



if __name__ == '__main__':
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Depth2CD(feature_exchange_type='le').to(device)
    xA = torch.randn(2, 3, 518, 518).to(device)
    xB = torch.randn(2, 3, 518, 518).to(device)
    output = model(xA, xB)
    print([o.shape for o in output])