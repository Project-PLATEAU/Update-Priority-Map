from dataclasses import dataclass, field
from typing import Dict, List

from omegaconf import DictConfig, OmegaConf

from .setting import ModelCD, ModelSR, Slicer, get_transform


@dataclass
class Configure:
    name: str = "configure.SRCDModelConfigure"


@dataclass
class Runner:
    name: str = "srcd.SRCDRunner"
    device: str = 'cuda'
    use_amp: bool = True
    norm_mean: List[float] = field(default_factory=lambda: [0.4914, 0.4822, 0.4465])
    norm_std: List[float] = field(default_factory=lambda: [0.2023, 0.1994, 0.2010])
    output: str = ''


@dataclass
class DataSet:
    name: str = "plateau.PlateauSRCD"
    data: str = "data/image"
    slicer_param: Slicer = Slicer()


@dataclass
class DataLoader:
    batch_size: int = 4
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = False


@dataclass
class Config:
    seed: int = 3407
    configure: Configure = Configure()
    runner: Runner = Runner()
    dataset: DataSet = DataSet()
    dataloader: DataLoader = DataLoader()
    model_sr: ModelSR = ModelSR()
    model_cd: ModelCD = ModelCD()
    transform: Dict = field(default_factory=dict)


def set_config(arguments: DictConfig) -> DictConfig:
    config = OmegaConf.structured(Config)

    config.dataset.data = arguments.image_path
    config.dataset.slicer_param.filelist = arguments.target
    config.dataset.slicer_param.width = arguments.patch_size
    config.dataset.slicer_param.height = arguments.patch_size
    config.dataset.slicer_param.sr_use = arguments.sr_use
    config.model_cd.length = arguments.patch_size
    config.dataloader.batch_size = arguments.batch_size
    config.model_cd.weight = arguments.cd_weight
    config.model_sr.weight = arguments.sr_weight
    config.model_sr.use = arguments.sr_use
    config.runner.output = arguments.output
    config.transform = get_transform()

    return config
