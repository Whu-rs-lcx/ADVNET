

from mmseg.registry import MODELS
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmseg.models.backbones.resnet import BasicBlock
import os


class EfficientCD(nn.Module):
    def __init__(self, neck=None, model_name='efficientnet_b5'):
        super(EfficientCD, self).__init__()
        # self.model = timm.create_model('efficientnet_b5', pretrained=True, features_only=True)
        self.model = timm.create_model('efficientnet_b5', pretrained=True, pretrained_cfg_overlay=dict(file=os.path.join(os.environ.get("PRETRAIN"), 'efficientnet_b5.sw_in12k_ft_in1k/pytorch_model.bin')), features_only=True)
        self.interaction_layers = ['blocks']
        # self.up_layer = [5, 3, 2, 1, 0]
        FPN_DICT = dict(
            type='FPN',
            in_channels=[24, 40, 64, 128, 176, 304, 512], # for b5
            out_channels=128,
            num_outs=7
        )
        norm_cfg = dict(type='SyncBN', requires_grad=True)

        # FPN_DICT = {'type': 'FPN', 'in_channels': [16, 24, 40, 80, 112, 192, 320], 'out_channels': 128, 'num_outs': 7}
        # FPN_DICT['in_channels'] = [i*2 for i in FPN_DICT['in_channels']]
        self.fpnA = MODELS.build(FPN_DICT)
        self.fpnB = MODELS.build(FPN_DICT)

        self.decode_layers1 = nn.Sequential(
            nn.Identity(),
            BasicBlock(inplanes=FPN_DICT['out_channels']*2, planes=FPN_DICT['out_channels']*2, norm_cfg=norm_cfg),
            BasicBlock(inplanes=FPN_DICT['out_channels']*2, planes=FPN_DICT['out_channels']*2, norm_cfg=norm_cfg),
            BasicBlock(inplanes=FPN_DICT['out_channels']*2, planes=FPN_DICT['out_channels']*2, norm_cfg=norm_cfg),
            BasicBlock(inplanes=FPN_DICT['out_channels']*2, planes=FPN_DICT['out_channels']*2, norm_cfg=norm_cfg),
            BasicBlock(inplanes=FPN_DICT['out_channels']*2, planes=FPN_DICT['out_channels']*2, norm_cfg=norm_cfg),
            BasicBlock(inplanes=FPN_DICT['out_channels']*2, planes=FPN_DICT['out_channels']*2, norm_cfg=norm_cfg)
        )
        self.decode_layers2 = nn.Sequential(
            nn.Identity(),
            BasicBlock(inplanes=FPN_DICT['out_channels']*2, planes=FPN_DICT['out_channels']*2, norm_cfg=norm_cfg),
            BasicBlock(inplanes=FPN_DICT['out_channels']*2, planes=FPN_DICT['out_channels']*2, norm_cfg=norm_cfg),
            BasicBlock(inplanes=FPN_DICT['out_channels']*2, planes=FPN_DICT['out_channels']*2, norm_cfg=norm_cfg),
            BasicBlock(inplanes=FPN_DICT['out_channels']*2, planes=FPN_DICT['out_channels']*2, norm_cfg=norm_cfg),
            BasicBlock(inplanes=FPN_DICT['out_channels']*2, planes=FPN_DICT['out_channels']*2, norm_cfg=norm_cfg),
            nn.Identity()
        )

        self.decode_conv0 = nn.Sequential(
            nn.Conv2d(FPN_DICT['out_channels']*2, FPN_DICT['out_channels']*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(FPN_DICT['out_channels']*2),
            nn.ReLU(inplace=True)
        )
        self.conv_seg0 = nn.Conv2d(FPN_DICT['out_channels']*2, 2, kernel_size=1)

        self.decode_conv1 = nn.Sequential(
            nn.Conv2d(FPN_DICT['out_channels']*2, FPN_DICT['out_channels']*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(FPN_DICT['out_channels']*2),
            nn.ReLU(inplace=True)
        )
        self.conv_seg1 = nn.Conv2d(FPN_DICT['out_channels']*2, 2, kernel_size=1)

        self.decode_conv2 = nn.Sequential(
            nn.Conv2d(FPN_DICT['out_channels']*2, FPN_DICT['out_channels']*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(FPN_DICT['out_channels']*2),
            nn.ReLU(inplace=True)
        )
        self.conv_seg2 = nn.Conv2d(FPN_DICT['out_channels']*2, 2, kernel_size=1)

        self.decode_conv3 = nn.Sequential(
            nn.Conv2d(FPN_DICT['out_channels']*2, FPN_DICT['out_channels']*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(FPN_DICT['out_channels']*2),
            nn.ReLU(inplace=True)
        )
        self.conv_seg3 = nn.Conv2d(FPN_DICT['out_channels']*2, 2, kernel_size=1)

        self.decode_conv4 = nn.Sequential(
            nn.Conv2d(FPN_DICT['out_channels']*2, FPN_DICT['out_channels']*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(FPN_DICT['out_channels']*2),
            nn.ReLU(inplace=True)
        )
        self.conv_seg4 = nn.Conv2d(FPN_DICT['out_channels']*2, 2, kernel_size=1)

        self.decode_conv5 = nn.Sequential(
            nn.Conv2d(FPN_DICT['out_channels']*2, FPN_DICT['out_channels']*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(FPN_DICT['out_channels']*2),
            nn.ReLU(inplace=True)
        )
        self.conv_seg5 = nn.Conv2d(FPN_DICT['out_channels']*2, 2, kernel_size=1)

    def change_feature(self, x, y):
        i = 2
        for index in range(0, len(x), i):
            x[index], y[index] = y[index], x[index]
        return x, y
    
    def euclidean_distance(self, img1, img2):
        diff_squared = (img1 - img2) ** 2
        distances = torch.sqrt(torch.sum(diff_squared, dim=1)).unsqueeze(1)
        max_distance = torch.max(distances)
        normalized_distances = torch.sigmoid(distances / max_distance)
        return normalized_distances

    def forward(self, xA, xB):
        for name, module in self.model.named_children():
            # print(f"Module Name: {name}")
            if name not in self.interaction_layers:
                xA = module(xA)
                xB = module(xB)
            else:
                xA_list = []
                xB_list = []
                for sub_name, sub_module in module.named_children():
                    # print(f"Module Name: {name}, Submodule Name: {sub_name}")
                    xA = sub_module(xA)
                    xB = sub_module(xB)
                    xA_list.append(xA)
                    xB_list.append(xB)
                break
        xA_list, xB_list = self.change_feature(xA_list, xB_list)
        xA_list = self.fpnA(xA_list)
        xB_list = self.fpnB(xB_list)
        xA_list, xB_list = self.change_feature(list(xA_list), list(xB_list))

        change_map = []
        curAB6 = torch.cat([xA_list[6], xB_list[6]], dim=1)
        curAB6 = self.euclidean_distance(xA_list[6], xB_list[6])*self.decode_layers1[6](curAB6)
        change_map.append(curAB6)

        curAB5 = torch.cat([xA_list[5], xB_list[5]], dim=1)
        curAB5 = curAB6+self.decode_layers1[5](curAB5)
        curAB5 = F.interpolate(curAB5, scale_factor=2, mode='bilinear', align_corners=False)
        dist5 = self.euclidean_distance(xA_list[5], xB_list[5])
        dist5 = F.interpolate(dist5, scale_factor=2, mode='bilinear', align_corners=False)
        curAB5 = dist5*self.decode_layers2[5](curAB5)
        change_map.append(curAB5)

        curAB4 = torch.cat([xA_list[4], xB_list[4]], dim=1)
        curAB4 = curAB5+self.decode_layers1[4](curAB4)
        curAB4 = self.euclidean_distance(xA_list[4], xB_list[4])*self.decode_layers2[4](curAB4)
        change_map.append(curAB4)

        curAB3 = torch.cat([xA_list[3], xB_list[3]], dim=1)
        curAB3 = curAB4+self.decode_layers1[3](curAB3)
        curAB3 = F.interpolate(curAB3, scale_factor=2, mode='bilinear', align_corners=
                                False)
        dist3 = self.euclidean_distance(xA_list[3], xB_list[3])
        dist3 = F.interpolate(dist3, scale_factor=2, mode='bilinear', align_corners=False)
        curAB3 = dist3*self.decode_layers2[3](curAB3)
        change_map.append(curAB3)

        curAB2 = torch.cat([xA_list[2], xB_list[2]], dim=1)
        curAB2 = curAB3+self.decode_layers1[2](curAB2)
        curAB2 = F.interpolate(curAB2, scale_factor=2, mode='bilinear', align_corners=
                                False)
        dist2 = self.euclidean_distance(xA_list[2], xB_list[2])
        dist2 = F.interpolate(dist2, scale_factor=2, mode='bilinear', align_corners=False)
        curAB2 = dist2*self.decode_layers2[2](curAB2)
        change_map.append(curAB2)

        curAB1 = torch.cat([xA_list[1], xB_list[1]], dim=1)
        curAB1 = curAB2+self.decode_layers1[1](curAB1)
        curAB1 = F.interpolate(curAB1, scale_factor=2, mode='bilinear', align_corners=
                                False)
        dist1 = self.euclidean_distance(xA_list[1], xB_list[1])
        dist1 = F.interpolate(dist1, scale_factor=2, mode='bilinear', align_corners=False)
        curAB1 = dist1*self.decode_layers2[1](curAB1)
        change_map.append(curAB1)
        

        change_map[0] = self.decode_conv0(change_map[0])
        change_map[0] = F.interpolate(change_map[0], size=(256, 256), mode='bilinear', align_corners=False)
        change_map[0] = self.conv_seg0(change_map[0])

        change_map[1] = self.decode_conv1(change_map[1])
        change_map[1] = F.interpolate(change_map[1], size=(256, 256), mode='bilinear', align_corners=False)
        change_map[1] = self.conv_seg1(change_map[1])

        change_map[2] = self.decode_conv2(change_map[2])
        change_map[2] = F.interpolate(change_map[2], size=(256, 256), mode='bilinear', align_corners=False)
        change_map[2] = self.conv_seg2(change_map[2])

        change_map[3] = self.decode_conv3(change_map[3])
        change_map[3] = F.interpolate(change_map[3], size=(256, 256), mode='bilinear', align_corners=False)
        change_map[3] = self.conv_seg3(change_map[3])

        change_map[4] = self.decode_conv4(change_map[4])
        change_map[4] = F.interpolate(change_map[4], size=(256, 256), mode='bilinear', align_corners=False)
        change_map[4] = self.conv_seg4(change_map[4])

        change_map[5] = self.decode_conv5(change_map[5])
        change_map[5] = F.interpolate(change_map[5], size=(256, 256), mode='bilinear', align_corners=False)
        change_map[5] = self.conv_seg5(change_map[5])

        return change_map


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    neck=dict(
        type='FPN',
        in_channels=[24, 40, 64, 128, 176, 304, 512], # for b5
        out_channels=128,
        num_outs=7)
    net = EfficientCD(neck=neck).to(device)
    xA = torch.randn(1, 3, 256, 256).to(device)
    xB = torch.randn(1, 3, 256, 256).to(device)
    out = net(xA, xB)
    print(out[0].shape)