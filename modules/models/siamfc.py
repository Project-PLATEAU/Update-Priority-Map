import torch
import torch.nn as nn
from omegaconf import DictConfig

from .misc.mixin import ModelMixin


class SiamDiff(nn.Module, ModelMixin):

    def __init__(self, config: DictConfig) -> None:
        super().__init__()

        self.encoder = self.get_encoder(**dict(name=config.encoder.name, config=config.encoder))
        self.decoder = self.get_decoder(**dict(name=config.decoder.name, config=config.decoder))
        self.head = nn.LazyConv2d(1, kernel_size=1)
        self.upsampler = nn.Upsample(size=config.length, mode='bilinear', align_corners=True)

    def forward(self, x_old: torch.Tensor, x_new: torch.Tensor) -> torch.Tensor:
        x_old_res = self.encoder(x_old)
        x_new_res = self.encoder(x_new)
        diff = [x_n - x_o for x_n, x_o in zip(x_new_res, x_old_res)]
        x_dec = self.decoder(diff[-1], diff[:-1])[-1]
        return self.upsampler(self.head(x_dec)).squeeze(dim=1)
