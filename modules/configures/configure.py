from __future__ import annotations

import random
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

from ..datasets.datamodule import DataModule
from ..runner.runner import ModelRunner
from ..utils.decorator import call_module

MODULES = 'modules'


class Configure(ABC):

    @classmethod
    @abstractmethod
    def configure(cls, config: DictConfig):
        pass

    @staticmethod
    def set_seeds(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    @staticmethod
    @call_module(f"{MODULES}.datasets")
    def configure_datamodule(instance: DataModule, config: DictConfig, **kwargs) -> DataModule:
        return instance(config)

    @staticmethod
    @call_module(f"{MODULES}.models")
    def configure_model(instance: torch.nn.Module, config: DictConfig, **kwargs) -> torch.nn.Module:
        model = instance(config)

        # Load model weight
        if bool(config.weight):
            model.load_state_dict(torch.load(config.weight)['model'])

        model.to(torch.device(config.device) if torch.cuda.is_available() else 'cpu')
        return model

    @staticmethod
    @call_module(f"{MODULES}.runner")
    def configure_runner(instance: ModelRunner, **kwargs) -> ModelRunner:
        return instance(**kwargs)


class SRCDModelConfigure(Configure):

    @classmethod
    def configure(cls, config: DictConfig) -> ModelRunner:
        cls.set_seeds(config.seed)
        if config.model_sr.use:
            model_sr = cls.configure_model(**dict(name=config.model_sr.name, config=config.model_sr))
        else:
            model_sr = nn.Identity()
        model_cd = cls.configure_model(**dict(name=config.model_cd.name, config=config.model_cd))
        datamodule = cls.configure_datamodule(**dict(name=config.dataset.name, config=config))
        return cls.configure_runner(model_sr=model_sr, model_cd=model_cd, datamodule=datamodule, **config.runner)


@call_module(f"{MODULES}.configures")
def get_trainer(instance: Configure, config: DictConfig, **kwargs) -> ModelRunner:
    return instance.configure(config)
