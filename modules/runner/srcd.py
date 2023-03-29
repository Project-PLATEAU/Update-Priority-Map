from pathlib import Path
from typing import Union

import torch
from kornia.enhance import Normalize
from omegaconf import OmegaConf
from rich.progress import track
from torch.utils.data import DataLoader

from .runner import ModelRunner


class SRCDRunner(ModelRunner):

    def configure_predict(self, **kwargs) -> dict:
        self.datamodule.configure('predict')
        self.slicer = self.datamodule.predict_dataset.slicer
        return dict(logger=self.configure_logger(self.__class__.__name__, log_filename="log/predict.log", stdout=True),
                    predict_dataloader=self.datamodule.predict_dataloader())

    def predict_epoch(self, dataloader: DataLoader) -> list[torch.Tensor]:
        predictions = []

        self.model_sr.eval()
        self.model_cd.eval()
        for batch in track(dataloader, description="Predict loop...", transient=False):
            predictions.append(self.predict_step(batch))
        return predictions

    @torch.inference_mode()
    def predict_step(self, batch: Union[torch.Tensor, list[torch.Tensor]]) -> torch.Tensor:
        data = self.set_device(batch, self.device)
        preds = self.forward(data)
        return self.set_device(preds[1], 'cpu', non_blocking=False)

    def forward(self, data: Union[torch.Tensor, list[torch.Tensor]]) -> Union[torch.Tensor, list[torch.Tensor]]:
        image = self.model_sr(self.set_device(data[1], self.device))
        norm_image = Normalize(OmegaConf.to_object(self.norm_mean), OmegaConf.to_object(self.norm_std))(image)
        return [image, self.model_cd(*self.set_device([data[0], norm_image], self.device))]

    def postprocess(self, predictions: list[torch.Tensor]) -> None:
        Path(self.output).mkdir(parents=True, exist_ok=True)
        self.slicer.save(torch.cat(predictions).sigmoid().numpy(), self.output)
