import timm

import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS

import timm
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from change_detection.utils.decode_block import ResDecodeBlock
from change_detection.utils.exchange import FeatureExchanger, ExchangeType
import random


ALL_MODES = [
    ExchangeType.LAYER,
    ExchangeType.RAND_LAYER,
    ExchangeType.CHANNEL,
    ExchangeType.RAND_CHANNEL,
    ExchangeType.SPATIAL,
    ExchangeType.RAND_SPATIAL,
]

FIXED_MODES = [
    ExchangeType.LAYER,
    ExchangeType.CHANNEL,
    ExchangeType.SPATIAL
]

RAND_MODES = [
    ExchangeType.RAND_LAYER,
    ExchangeType.RAND_CHANNEL,
    ExchangeType.RAND_SPATIAL
]


class SEED_EfficientNet(nn.Module):
    def __init__(self, neck=None, exchange_mode='le'):
        super(SEED_EfficientNet, self).__init__()
        try:
            self.model = timm.create_model('efficientnet_b4', pretrained=True, pretrained_cfg_overlay=dict(file='pretrained/efficientnet_b4.ra2_in1k/pytorch_model.bin'), features_only=True)
        except:
            self.model = timm.create_model('efficientnet_b4', pretrained=True, features_only=True)
        '''
        '''
        FPN_DICT = {'type': 'FPN', 'in_channels': [32, 56, 160, 448], 'out_channels': 256, 'num_outs': 4}
        self.fpn = MODELS.build(FPN_DICT)

        self.decode_layersA = nn.Sequential(
            nn.Identity(),
            ResDecodeBlock(in_channels=FPN_DICT['out_channels']),
            ResDecodeBlock(in_channels=FPN_DICT['out_channels']),
            ResDecodeBlock(in_channels=FPN_DICT['out_channels']),
            ResDecodeBlock(in_channels=FPN_DICT['out_channels'])
        )

        self.sigmoid = nn.Sigmoid()

        self.ex_func = FeatureExchanger(training=self.training)

        self.decode_conv = nn.Sequential(
            nn.Conv2d(FPN_DICT['out_channels'], FPN_DICT['out_channels'], kernel_size=3, padding=1),
            nn.BatchNorm2d(FPN_DICT['out_channels']),
            nn.ReLU(inplace=True)
        )
        self.conv_seg = nn.Conv2d(FPN_DICT['out_channels'], 2, kernel_size=1)

    def decode_stage(self, feature_list):
        x1, x2, x3, x4 = feature_list
        x4 = self.decode_layersA[4](x4)
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x3 = x3 + x4
        x3 = self.decode_layersA[3](x3)
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x2 = x2 + x3
        x2 = self.decode_layersA[2](x2)
        x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
        x1 = x1 + x2
        x1 = self.decode_layersA[1](x1)
        return x1

    def decode_head(self, x):
        x = self.decode_conv(x)
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        x = self.conv_seg(x)
        return x

    def forward(self, xA, xB):
        xA_list = self.model(xA)[1:]
        xB_list = self.model(xB)[1:]
        if self.training:
            # For training, randomly choose an exchange mode
            xA_list, xB_list = self.ex_func.exchange(xA_list, xB_list, mode=random.choice(FIXED_MODES), thresh=0.5)
        else:
            # For inference, use a fixed exchange mode
            xA_list, xB_list = self.ex_func.exchange(xA_list, xB_list, mode=ExchangeType.LAYER, thresh=0.5)

        xA_list = self.fpn(xA_list)
        xB_list = self.fpn(xB_list)

        outA = self.decode_stage(xA_list)
        outB = self.decode_stage(xB_list)

        outA = self.decode_head(outA)
        outB = self.decode_head(outB)

        return outA, outB
