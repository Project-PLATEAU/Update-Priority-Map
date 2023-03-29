from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from .layer.rrdb import RRDB
from .misc.mixin import ModelMixin


class RRDBNet(nn.Module, ModelMixin):

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.first = nn.LazyConv2d(config.residual_blocks.block_param.in_channels, kernel_size=3, stride=1, padding=1)
        self.residual_blocks = RRDB(**OmegaConf.to_container(config.residual_blocks))
        self.pre_add = nn.Conv2d(config.residual_blocks.block_param.in_channels,
                                 config.residual_blocks.block_param.in_channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.upsamples = self.create_upsample(config.upsample)
        self.head = self.create_head(config.head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = self.first(x)
        x_res = self.pre_add(self.residual_blocks(x_f))
        x_add = torch.add(x_f, x_res)
        return self.head(self.upsamples(x_add)).sigmoid()

    def create_upsample(self, config: DictConfig) -> nn.Sequential:
        config.conv.parameters.in_channels = config.in_channels
        config.conv.parameters.out_channels = config.in_channels
        if config.up.name == 'PixelShuffle':
            config.conv.parameters.out_channels = config.in_channels * config.up.parameters.upscale_factor**2

        output = nn.Sequential()
        for i in range(config.num_blocks):
            output.add_module(
                f'upsample{i + 1:02}',
                nn.Sequential(self.get_layer(**OmegaConf.to_container(config.conv)),
                              self.get_layer(**OmegaConf.to_container(config.up)),
                              self.get_layer(**OmegaConf.to_container(config.act))))
        return output

    def create_head(self, config: DictConfig) -> nn.Sequential:
        config.conv.parameters.in_channels = config.in_channels
        config.conv.parameters.out_channels = config.in_channels

        output = nn.Sequential()
        for i in range(config.num_blocks):
            output.add_module(
                f'head{i + 1:02}',
                nn.Sequential(
                    self.get_layer(**OmegaConf.to_container(config.conv)),
                    self.get_norm_layer(config.in_channels, config.norm) if not config.norm == {} else nn.Identity(),
                    self.get_layer(**OmegaConf.to_container(config.act))))
        output.add_module('last', nn.LazyConv2d(config.out_channels, kernel_size=3, stride=1, padding=1))
        return output
