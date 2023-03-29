import torch
from torch.utils.data import Dataset

from ..inference.tiled import Slicer
from ..transforms.compose import IntegrateCompose
from .datamodule import DataModule


class PlateauSRCD(DataModule):

    def configure(self, key: str) -> None:
        if key == 'predict':
            self.predict_dataset = PlateauCDPredictDataset(root=self.conf_dataset.data,
                                                           slicer_param=self.conf_dataset.slicer_param,
                                                           transforms=self.get_transforms('predict'))
        else:
            raise ValueError(f"{key} is not implemented.")


class PlateauCDPredictDataset(Dataset):

    def __init__(self, root: str, slicer_param: dict, transforms: IntegrateCompose) -> None:
        self.root = root
        self.transforms = transforms

        slicer_param['directory'] = f"{self.root}/old"
        self.slicer = Slicer(**slicer_param)

        if slicer_param.sr_use:
            self.scale = 0.5
        else:
            self.scale = 1.0

    def __len__(self) -> int:
        return len(self.slicer)

    def __getitem__(self, idx: int) -> tuple[tuple[torch.Tensor]]:
        # Read image
        old_image = self.slicer.cut(idx, f"{self.root}/old")
        new_image = self.slicer.cut(idx, f"{self.root}/new", scale=self.scale)

        # Transform
        augmented = self.transforms(**dict(image=old_image, image0=new_image))
        return (augmented['image'], augmented['image0'])
