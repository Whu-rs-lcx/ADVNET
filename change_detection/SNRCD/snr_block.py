import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F


class ChannelGate_sub(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""

    def __init__(self, in_channels, num_gates=None, return_gates=False,
                 gate_activation='sigmoid', reduction=16, layer_norm=False):
        super(ChannelGate_sub, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels//reduction, kernel_size=1, bias=True, padding=0)
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm((in_channels//reduction, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels//reduction, num_gates, kernel_size=1, bias=True, padding=0)
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError("Unknown gate activation: {}".format(gate_activation))

    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input * x, input * (1 - x), x


class SNR_Block(nn.Module):
    def __init__(self, in_channels):
        super(SNR_Block, self).__init__()
        self.style_reid_layer1 = ChannelGate_sub(in_channels, num_gates=in_channels, return_gates=False,
                 gate_activation='sigmoid', reduction=16, layer_norm=False)
        self.IN1 = nn.InstanceNorm2d(in_channels, affine=True)

    def forward(self, x):
        x_IN_1 = self.IN1(x)
        x_style_1 = x - x_IN_1
        x_style_1_reid_useful, x_style_1_reid_useless, selective_weight_useful_1 = self.style_reid_layer1(x_style_1)
        x_1 = x_IN_1 + x_style_1_reid_useful
        return x_1