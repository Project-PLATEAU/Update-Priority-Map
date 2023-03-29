from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Union

import backtrace
import torch

from ..loggers.logger import StandardLogger
from ..utils.timer import Timer

warnings.simplefilter('ignore', UserWarning)

backtrace.hook(reverse=True, strip_path=True)


class ModelRunner(ABC):

    def __init__(self, **kwargs) -> None:
        super().__init__()
        torch.backends.cudnn.benchmark = True
        for k, v in kwargs.items():
            setattr(self, k, v)

    @abstractmethod
    def postprocess(self, data: torch.Tensor) -> None:
        pass

    def configure(self, key: str, **kwargs) -> dict:
        return getattr(self, f'configure_{key}')(**kwargs)

    def predict(self) -> None:
        prepares = self.configure('predict')
        self.logger = prepares['logger']

        timer = Timer(device=self.device)
        self.postprocess(self.predict_epoch(prepares['predict_dataloader']))
        self.logger.log('info', timer.print("Done predicting", mode='total'))

    def configure_predict(self, **kwargs) -> dict:
        self.datamodule.configure('predict')
        return dict(logger=self.configure_logger(self.__class__.__name__,
                                                 log_filename="log/generate_probmap.log",
                                                 stdout=True),
                    predict_dataloader=self.datamodule.predict_dataloader())

    @staticmethod
    def set_device(data: Union[torch.Tensor, list[torch.Tensor]],
                   device: str,
                   non_blocking: bool = True) -> Union[torch.Tensor, list[torch.Tensor]]:
        if isinstance(data, torch.Tensor):
            data = data.to(device=device, non_blocking=non_blocking)
        else:
            data = [d.to(device=device, non_blocking=non_blocking) for d in data]
        return data

    @staticmethod
    def configure_logger(target: str, log_filename: str, stdout: bool = False) -> StandardLogger:
        logger = StandardLogger(log_filename=log_filename, stdout=stdout)
        logger.configure(target)
        return logger
