import torch
import torch.nn as nn

from ...utils.decorator import call_module


@call_module(["torch.nn", "timm.models.layers", "modules.models.layer"])
def get_layer(instance: nn.Module, **kwargs) -> nn.Module:
    if kwargs.get('parameters'):
        parameters = kwargs['parameters']
        return instance(*parameters) if isinstance(parameters, list) else instance(**parameters)
    else:
        return instance


# normal SE module
class ChannelSE(nn.Module):

    def __init__(self, in_channels: int, reduction_ratio: int, activate: nn.Module) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.layer = nn.Sequential(nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1), activate,
                                   nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.layer(self.pool(x))


class SpatialSE(nn.Module):

    def __init__(self, in_channels: int) -> torch.Tensor:
        super().__init__()
        self.layer = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.layer(x)


class SpatialChannelSE(nn.Module):

    def __init__(self, in_channels: int, reduction_ratio: int, activate: dict, method: str = 'max') -> None:
        super().__init__()
        self.cse = ChannelSE(in_channels, reduction_ratio, get_layer(**activate))
        self.sse = SpatialSE(in_channels)
        self.head = torch.add if method == 'add' else torch.mul if method == 'multiply' else torch.max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.cse(x), self.sse(x))
