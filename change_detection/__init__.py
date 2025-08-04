# 假设这段代码在 __init__.py 同级或包内的某个模块里

from .bit import BIT
from .cdnet import CDNet
from .dsifn import DSIFN
from .lunet import LUNet
from .p2v import P2VNet
from .stanet import STANet
from .siamunet_conc import SiamUNet_conc
from .siamunet_diff import SiamUNet_diff
from .mfpnet import MFPNET
from .sunet import SUNnet
from .CGNet import CGNet, HCGMNet
from .AFCF3D_Net import AFCF3D
from .elgcnet import ELGCNet
from .acabfnet import ACABFNet
from .dminet import DMINet
from .CDNeXt.cdnext import CDNeXt
from .GASNet import GASNet
from .hanet import HANet
from .isdanet import ISDANet
from .STRobustNet.STRobustNet import STRobustNet
from .ScratchFormer.scratch_former import ScratchFormer
from .DARNet.DARNet import DARNet
from .BASNet import BASNet
from .rctnet import BaseNet as RCTNet
from .seed_swint import SEED_SwinT
from .seed_efficientnet import SEED_EfficientNet
from .lenet import LENet
from .EfficientCD import EfficientCD
from .depth2cd import Depth2CD
from .SNRCD.SNRNet import SNRNet
from .CWCD.CWNet import CWNet_CNN
from .ADVNet.baseline import BaselineAdv as ADVNet
# 方法一：手动构建映射字典
_model_factory = {
    'BIT': BIT,
    'CDNet': CDNet,
    'DSIFN': DSIFN,
    'LUNet': LUNet,
    'P2VNet': P2VNet,
    'STANet': STANet,
    'SiamUNet_conc': SiamUNet_conc,
    'SiamUNet_diff': SiamUNet_diff,
    'MFPNET': MFPNET,
    'SUNnet': SUNnet,
    'CGNet': CGNet,
    'HCGMNet': HCGMNet,
    'AFCF3D': AFCF3D,
    'ELGCNet': ELGCNet,
    'ACABFNet': ACABFNet,
    'DMINet': DMINet,
    'CDNeXt': CDNeXt,
    'GASNet': GASNet,
    'HANet': HANet,
    'ISDANet': ISDANet,
    'STRobustNet': STRobustNet,
    'ScratchFormer': ScratchFormer,
    'DARNet': DARNet,
    'BASNet': BASNet,
    'RCTNet': RCTNet,
    'SEED_SwinT': SEED_SwinT,
    'SEED_EfficientNet': SEED_EfficientNet,
    'EfficientCD': EfficientCD,
    'LENet': LENet,
    'Depth2CD': Depth2CD,
    'SNRNet': SNRNet,
    'CWNet_CNN': CWNet_CNN,
    'ADVNet': ADVNet,
}

def build_model(name: str, *args, **kwargs):
    """
    根据 name 字符串返回对应的模型实例。
    支持传入构造函数的参数 args, kwargs。
    """
    try:
        ModelClass = _model_factory[name]
    except KeyError:
        raise ValueError(f"Unknown model name '{name}'. Available: {list(_model_factory.keys())}")
    return ModelClass(*args, **kwargs)
