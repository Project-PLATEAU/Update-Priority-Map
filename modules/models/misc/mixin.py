from __future__ import annotations

from dataclasses import InitVar, asdict, dataclass, field
from typing import Any

import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from ...utils.decorator import call_module


@dataclass
class Normal:
    mean: float = 0.
    std: float = 0.01


@dataclass
class Xavier:
    nonlinearity: InitVar[str] = 'relu'
    gain: float = field(init=False)

    def __post_init__(self, nonlinearity: str) -> None:
        self.gain = nn.init.calculate_gain(nonlinearity)


@dataclass
class Kaiming:
    mode: str = 'fan_out'
    nonlinearity: str = 'relu'


class ModelMixin:

    init_items = dict(normal=Normal, xavier=Xavier, kaiming=Kaiming)

    def initialize_parameters(self, params: Any, method: str = 'kaiming_normal', activation: str = 'relu') -> None:

        def recursive_initialization(p: Any, **kwargs) -> None:
            if any(hasattr(p, i) for i in ['weight', 'bias']):
                self.initialize(p, **kwargs)
            elif callable(p.children):
                for m in p.children():
                    recursive_initialization(m, **kwargs)

    def initialize(self, params: Any, method: str = 'kaiming_normal', activation: str = 'relu') -> None:
        initialize_method = getattr(nn.init, f'{method}_')
        initialize_item = self.init_items[method.split('_')[0]]
        initialize_item = initialize_item() if method == 'normal_' else initialize_item(nonlinearity=activation)

        if isinstance(params, (nn.modules.conv._ConvNd, nn.modules.conv._ConvTransposeNd)):
            initialize_method(params.weight, **asdict(initialize_item))
        elif isinstance(params, (nn.modules.batchnorm._BatchNorm, nn.GroupNorm, nn.LayerNorm)):
            nn.init.constant_(params.weight.data, 1)
        elif isinstance(params, nn.Linear):
            nn.init.normal_(params.weight, **asdict(Normal()))
        if params.bias is not None:
            nn.init.constant_(params.bias, 0)

    def get_norm_layer(self, channels: int, config: DictConfig) -> nn.Module:
        if config.name == 'GroupNorm':
            config.parameters.num_channels = channels
        elif config.name == 'LayerNorm2d':
            config.parameters.num_channels = channels
        elif not config.name[:4] == 'Lazy':
            config.parameters.num_features = channels
        return self.get_layer(**OmegaConf.to_container(config))

    @staticmethod
    @call_module(["torch.nn", "timm.models.layers", "modules.models.layer"])
    def get_layer(instance: nn.Module, **kwargs) -> nn.Module:
        if kwargs.get('parameters'):
            parameters = kwargs['parameters']
            return instance(*parameters) if isinstance(parameters, list) else instance(**parameters)
        else:
            return instance

    @staticmethod
    @call_module("modules.models")
    def get_encoder(instance: nn.Module, config: DictConfig, **kwargs) -> nn.Module:
        return instance(config)

    @staticmethod
    @call_module("modules.models")
    def get_decoder(instance: nn.Module, config: DictConfig, **kwargs) -> nn.Module:
        return instance(config)
