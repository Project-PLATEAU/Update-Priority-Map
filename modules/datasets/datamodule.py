from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any

from omegaconf import DictConfig
from torch.utils.data import DataLoader

from ..transforms.compose import IntegrateCompose
from ..utils.decorator import call_module


def worker_init(worker_id: int) -> None:
    random.seed(worker_id)


class DataModule(ABC):

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.conf_dataset = config.dataset
        self.conf_dataloader = config.dataloader
        self.conf_transforms = config.transform

    @abstractmethod
    def configure(self, key: str):
        pass

    def get_transforms(self, key: str) -> IntegrateCompose:
        config = self.conf_transforms[key] if not key == 'predict' else self.conf_transforms
        return IntegrateCompose([self.get_transform(**{'name': k, 'parameters': v}) for k, v in config.items()])

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.predict_dataset, **self.conf_dataloader)

    @staticmethod
    @call_module(["modules.transforms", 'albumentations', "torchvision.transforms"])
    def get_transform(instance: Any, parameters: dict[str, Any], **kwargs) -> Any:
        return instance(**parameters)
