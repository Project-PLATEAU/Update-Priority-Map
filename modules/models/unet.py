from __future__ import annotations

from copy import copy
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from .misc.mixin import ModelMixin


class EncoderBlock(nn.Module):

    def __init__(self, convs: list[nn.Module], norms: list[nn.Module], act: nn.Module, downsample: nn.Module,
                 se: nn.Module) -> None:
        super().__init__()
        self.downsample = downsample
        self.block0 = nn.Sequential(convs[0], norms[0], act)
        self.block1 = nn.Sequential(convs[1], norms[1], act)
        if not se == nn.Identity:
            self.block1.add_module('se', se)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x) if not self.downsample == nn.Identity else x
        x = self.block0(x)
        return self.block1(x)


class DecoderBlock(nn.Module):

    def __init__(self, convs: list[nn.Module], norms: list[nn.Module], act: nn.Module, upsample: nn.Module,
                 se: nn.Module) -> None:
        super().__init__()
        self.upsample = upsample
        self.block0 = nn.Sequential(convs[0], norms[0], act)
        self.block1 = nn.Sequential(convs[1], norms[1], act)
        if not se == nn.Identity:
            self.block1.add_module('se', se)

    def forward(self, x: torch.Tensor, x_enc: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, x_enc], dim=1) if x_enc is not None else x
        x = self.block0(x)
        return self.block1(x)


class UNetEncoder(nn.Module, ModelMixin):

    def __init__(self, config: DictConfig, in_channels: int = 3) -> None:
        super().__init__()
        in_channels = [in_channels] + config.out_channels[:-1]
        out_channels = config.out_channels

        self.blocks = nn.ModuleList([
            self.make_block(i, in_c, out_c, copy(config))
            for i, (in_c, out_c) in enumerate(zip(in_channels, out_channels))
        ])

        self.initialize_parameters(self.blocks)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        output = []
        for block in self.blocks:
            x = block(x)
            output.append(x)
        return output

    def __len__(self) -> int:
        return len(self.blocks)

    def make_block(self, idx: int, in_c: int, out_c: int, params: DictConfig) -> EncoderBlock:
        # Set conv parameters
        params.conv.parameters.in_channels = in_c
        params.conv.parameters.out_channels = out_c

        # Set downsample
        if idx == 0:
            params.downsample.name = 'Identity'
            params.downsample.parameters = dict()

        # Set SE Layer
        if params.se == {}:
            params.se.name = 'Identity'
            params.se.parameters = dict()
        else:
            params.se.parameters.in_channels = out_c

        conv = self.get_layer(**OmegaConf.to_container(params.conv))
        norm = self.get_norm_layer(out_c, params.norm)
        params.conv.parameters.in_channels = out_c
        return EncoderBlock([conv, self.get_layer(**OmegaConf.to_container(params.conv))],
                            [norm, self.get_layer(**OmegaConf.to_container(params.norm))],
                            self.get_layer(**OmegaConf.to_container(params.act)),
                            self.get_layer(**OmegaConf.to_container(params.downsample)),
                            self.get_layer(**OmegaConf.to_container(params.se)))


class UNetDecoder(nn.Module, ModelMixin):

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        out_channels = config.out_channels

        self.blocks = nn.ModuleList([self.make_block(out_c, copy(config)) for out_c in out_channels])

        self.initialize_parameters(self.blocks)

    def forward(self, x: torch.Tensor, x_enc: list[torch.Tensor]) -> list[torch.Tensor]:
        output = []
        for block, enc in zip(self.blocks, reversed(x_enc)):
            x = block(x, enc)
            output.append(x)
        return output

    def __len__(self) -> int:
        return len(self.blocks)

    def make_block(self, out_c: int, params: DictConfig) -> EncoderBlock:
        # Set conv parameters
        params.conv.name = 'Lazy' + params.conv.name if not params.conv.name[:4] == 'Lazy' else params.conv.name
        params.conv.parameters.out_channels = out_c

        # Set SE Layer
        if params.se == {}:
            params.se.name = 'Identity'
            params.se.parameters = dict()
        else:
            params.se.parameters.in_channels = out_c

        conv = self.get_layer(**OmegaConf.to_container(params.conv))
        norm = self.get_norm_layer(out_c, params.norm)
        return DecoderBlock([conv, self.get_layer(**OmegaConf.to_container(params.conv))],
                            [norm, self.get_layer(**OmegaConf.to_container(params.norm))],
                            self.get_layer(**OmegaConf.to_container(params.act)),
                            self.get_layer(**OmegaConf.to_container(params.upsample)),
                            self.get_layer(**OmegaConf.to_container(params.se)))


class UNet(nn.Module, ModelMixin):

    def __init__(self, config: DictConfig) -> None:
        super().__init__()

        self.encoder = self.get_encoder(**dict(name=config.encoder.name, config=config.encoder))
        self.decoder = self.get_decoder(**dict(name=config.decoder.name, config=config.decoder))
        self.head = nn.LazyConv2d(out_channels=config.out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_enc = self.encoder(x)
        x_dec = self.decoder(x_enc[-1], x_enc[:-1])
        return self.head(x_dec[-1])
