from mmseg.registry import MODELS

import timm
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from change_detection.utils.decode_block import *
from .snr_block import SNR_Block
from change_detection.utils.exchange import FeatureExchanger, ExchangeType
import torch

from typing import Tuple, Union
import os



class SNRNet(nn.Module):
    def __init__(self, neck=None, feature_exchange_type='le'):
        super(SNRNet, self).__init__()
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
        self.decode_conv = nn.Sequential(
            nn.Conv2d(FPN_DICT['out_channels'], FPN_DICT['out_channels'], kernel_size=3, padding=1),
            nn.BatchNorm2d(FPN_DICT['out_channels']),
            nn.ReLU(inplace=True)
        )
        self.conv_seg = nn.Conv2d(FPN_DICT['out_channels'], 2, kernel_size=1)

        self.snr_block = nn.Sequential(
            SNR_Block(in_channels=FPN_DICT['in_channels'][0]),
            SNR_Block(in_channels=FPN_DICT['in_channels'][1]),
            SNR_Block(in_channels=FPN_DICT['in_channels'][2]),
            SNR_Block(in_channels=FPN_DICT['in_channels'][3])
        )

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
        
        xA_list = [self.snr_block[i](xA_list[i]) for i in range(len(xA_list))]
        xB_list = [self.snr_block[i](xB_list[i]) for i in range(len(xB_list))]

        xA_list, xB_list = self.ex_func.exchange(xA_list, xB_list, mode=ExchangeType.LAYER, thresh=0.5)

        xA_list = self.fpnA(xA_list)
        xB_list = self.fpnA(xB_list)

        xA_list = [x.permute(0, 2, 3, 1) for x in xA_list]
        xB_list = [x.permute(0, 2, 3, 1) for x in xB_list]
        outA = self.decode_stage(xA_list)
        outB = self.decode_stage(xB_list)

        outA = self.decode_head(outA)
        outB = self.decode_head(outB)
        return outA, outB



class SNRNet_ResNet(nn.Module):
    def __init__(self, neck=None, feature_exchange_type='le'):
        super(SNRNet_ResNet, self).__init__()
        try:
            self.model = timm.create_model('resnet50', pretrained=True, pretrained_cfg_overlay=dict(file='pretrained/resnet50.a1_in1k/pytorch_model.bin'), features_only=True)
        except:
            self.model = timm.create_model('resnet50', pretrained=True, features_only=True)
        '''
        swinv2_base_window8_256
        torch.Size([16, 64, 128, 128])
        torch.Size([16, 256, 64, 64])
        torch.Size([16, 512, 32, 32])
        torch.Size([16, 1024, 16, 16])
        torch.Size([16, 2048, 8, 8])
        '''
        FPN_DICT = {'type': 'FPN', 'in_channels': [256, 512, 1024, 2048], 'out_channels': 256, 'num_outs': 4}
        self.fpnA = MODELS.build(FPN_DICT)
        # self.fpnB = MODELS.build(FPN_DICT)

        self.decode_layersA = nn.Sequential(
            nn.Identity(),
            ResDecodeBlock(in_channels=FPN_DICT['out_channels']),
            ResDecodeBlock(in_channels=FPN_DICT['out_channels']),
            ResDecodeBlock(in_channels=FPN_DICT['out_channels']),
            ResDecodeBlock(in_channels=FPN_DICT['out_channels'])
        )
        self.feature_exchange_type = feature_exchange_type
        self.sigmoid = nn.Sigmoid()

        self.snr_block = nn.Sequential(
            SNR_Block(in_channels=FPN_DICT['in_channels'][0]),
            SNR_Block(in_channels=FPN_DICT['in_channels'][1]),
            SNR_Block(in_channels=FPN_DICT['in_channels'][2]),
            SNR_Block(in_channels=FPN_DICT['in_channels'][3])
        )

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

        xA_list = [self.snr_block[i](xA_list[i]) for i in range(len(xA_list))]
        xB_list = [self.snr_block[i](xB_list[i]) for i in range(len(xB_list))]

        xA_list, xB_list = self.ex_func.exchange(xA_list, xB_list, mode=ExchangeType.LAYER, thresh=0.5)

        xA_list = self.fpnA(xA_list)
        xB_list = self.fpnA(xB_list)

        outA = self.decode_stage(xA_list)
        outB = self.decode_stage(xB_list)
        outA = self.decode_head(outA)
        outB = self.decode_head(outB)
        change_maps = [outA, outB]
        return change_maps
    

if __name__ == '__main__':
    model = SNRNet()
    xA = torch.randn(2, 3, 256, 256)
    xB = torch.randn(2, 3, 256, 256)
    outA, outB = model(xA, xB)
    print(outA.shape, outB.shape)  # Should print torch.Size([2, 2, 256, 256]) for both outputs
    print("Model forward pass successful.")