from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from ..misc.mixin import ModelMixin


class DenseResidualBlock(nn.Module, ModelMixin):

    def __init__(self, in_channels: int, middle_channels: int, num_blocks: int, weight: float,
                 block_param: dict[str, Any]) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor([weight]))
        for i in range(num_blocks):
            in_c = in_channels + i * middle_channels
            out_c = middle_channels if i < num_blocks - 1 else in_channels
            self.add_module(f'block{i + 1:02}', self.create_block(in_c, out_c, OmegaConf.create(block_param)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for i, block in enumerate(self.blocks):
            out = block(residual)
            residual = torch.cat([residual, out], dim=1) if i < len(self.blocks) - 1 else out
        return x + self.weight * residual

    def create_block(self, in_channels: int, out_channels: int, config: DictConfig) -> nn.Sequential:
        # convolution
        config.conv.parameters.in_channels = in_channels
        config.conv.parameters.out_channels = out_channels
        return nn.Sequential(self.get_layer(**OmegaConf.to_container(config.conv)),
                             self.get_norm_layer(out_channels, config.norm) if not config.norm == {} else nn.Identity(),
                             self.get_layer(**OmegaConf.to_container(config.act)))

    @property
    def blocks(self) -> list[nn.Module]:
        idx = 1
        layers = []
        while hasattr(self, f'block{idx:02}'):
            layers.append(self.__getattr__(f'block{idx:02}'))
            idx += 1
        return layers


# Residual in Residual Dense Block
class RRDB(nn.Module):

    def __init__(self, num_blocks: int, weight: float, block_param: dict[str, Any]) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor([weight]))
        for i in range(num_blocks):
            self.add_module(f'dense_block{i + 1:02}', DenseResidualBlock(**block_param))

    def forward(self, x):
        residual = x
        for block in self.dense_blocks:
            residual = block(residual)
        return x + self.weight * residual

    @property
    def dense_blocks(self) -> list[nn.Module]:
        idx = 1
        layers = []
        while hasattr(self, f'dense_block{idx:02}'):
            layers.append(self.__getattr__(f'dense_block{idx:02}'))
            idx += 1
        return layers
