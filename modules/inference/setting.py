from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Slicer:
    filelist: str = ''
    directory: str = ''
    height: int = 1024
    width: int = 1024
    horizontal_overlap: float = 2.
    vertical_overlap: float = 2.
    sr_use: bool = True


@dataclass
class Layer:
    name: str = ''
    parameters: Dict = field(default_factory=dict)


@dataclass
class BlockParam:
    conv: Layer = Layer(name='Conv2d', parameters=dict(kernel_size=3, stride=1, padding=1))
    norm: Dict = field(default_factory=dict)
    act: Layer = Layer(name='SiLU', parameters=dict(inplace=True))


@dataclass
class Block:
    num_blocks: int = 5
    weight: float = 0.2
    in_channels: int = 64
    middle_channels: int = 32
    block_param: BlockParam = BlockParam()


@dataclass
class Encoder:
    name: str = "unet.UNetEncoder"
    out_channels: List[int] = field(default_factory=lambda: [64, 64, 128, 256, 512])
    conv: Layer = Layer(name='Conv2d', parameters=dict(in_channels=3, out_channels=64, kernel_size=3, padding=1))
    norm: Layer = Layer(name='BatchNorm2d', parameters=dict(num_features=64))
    act: Layer = Layer(name='SiLU', parameters=dict(inplace=True))
    se: Layer = Layer(name="seblock.SpatialChannelSE",
                      parameters=dict(in_channels=64,
                                      reduction_ratio=2,
                                      activate=dict(name='SiLU', parameters=dict(inplace=True)),
                                      method='max'))
    downsample: Layer = Layer(name='MaxPool2d', parameters=dict(kernel_size=2, stride=2))


@dataclass
class Decoder:
    name: str = "unet.UNetDecoder"
    out_channels: List[int] = field(default_factory=lambda: [512, 256, 128, 64])
    conv: Layer = Layer(name='LazyConv2d', parameters=dict(out_channels=512, kernel_size=3, padding=1))
    norm: Layer = Layer(name='BatchNorm2d', parameters=dict(num_features=256))
    act: Layer = Layer(name='SiLU', parameters=dict(inplace=True))
    se: Layer = Layer(name="seblock.SpatialChannelSE",
                      parameters=dict(in_channels=64,
                                      reduction_ratio=2,
                                      activate=dict(name='SiLU', parameters=dict(inplace=True)),
                                      method='max'))
    upsample: Layer = Layer(name='Upsample', parameters=dict(scale_factor=2, mode='bilinear', align_corners=True))


@dataclass
class ResidualBlocks:
    num_blocks: int = 69
    weight: float = 0.2
    block_param: Block = Block()


@dataclass
class Upsample:
    num_blocks: int = 1
    in_channels: int = 64
    conv: Layer = Layer(name='Conv2d', parameters=dict(kernel_size=3, stride=1, padding=1))
    act: Layer = Layer(name='SiLU', parameters=dict(inplace=True))
    up: Layer = Layer(name='PixelShuffle', parameters=dict(upscale_factor=2))


@dataclass
class Head:
    num_blocks: int = 1
    in_channels: int = 64
    out_channels: int = 3
    conv: Layer = Layer(name='Conv2d', parameters=dict(kernel_size=3, stride=1, padding=1))
    norm: Dict = field(default_factory=dict)
    act: Layer = Layer(name='SiLU', parameters=dict(inplace=True))


@dataclass
class Normalize:
    mean: List[float] = field(default_factory=lambda: [0.4914, 0.4822, 0.4465])
    std: List[float] = field(default_factory=lambda: [0.2023, 0.1994, 0.2010])
    max_pixel_value: float = 255.


@dataclass
class Runner:
    name: str = "srcd.SRCDRunner"
    device: str = 'cuda'
    use_amp: bool = True
    norm_mean: List[float] = field(default_factory=lambda: [0.4914, 0.4822, 0.4465])
    norm_std: List[float] = field(default_factory=lambda: [0.2023, 0.1994, 0.2010])
    output: str = ''


@dataclass
class ModelSR:
    name: str = "esrnet.RRDBNet"
    in_channels: int = 3
    out_channels: int = 3
    device: str = 'cuda'
    residual_blocks: ResidualBlocks = ResidualBlocks()
    upsample: Upsample = Upsample()
    head: Head = Head()
    weight: str = "pretrained/rrdb.pth"
    use: bool = True


@dataclass
class ModelCD:
    name: str = "siamfc.SiamDiff"
    in_channels: int = 3
    out_channels: int = 2
    length: int = 512
    device: str = 'cuda'
    encoder: Encoder = Encoder()
    decoder: Decoder = Decoder()
    weight: str = "pretrained/unet.pth"


def get_transform():
    return {
        "autocontrast.AutoContrast": dict(always_apply=True, cutoff=2., invalid=0),
        "change_detection.NormalizeCD": Normalize()
    }
