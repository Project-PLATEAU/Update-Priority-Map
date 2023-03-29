from typing import Union

import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform
from PIL import Image, ImageOps


class AutoContrast(ImageOnlyTransform):

    def __init__(self,
                 cutoff: float = 2.,
                 invalid: int = 0,
                 p: float = 0.5,
                 always_apply: bool = False,
                 **kwargs) -> None:
        super(AutoContrast, self).__init__(always_apply=always_apply, p=p)
        self.cutoff = cutoff
        self.invalid = invalid

    def apply(self, img: Union[np.ndarray, Image.Image], **kwargs: dict) -> Union[np.ndarray, Image.Image]:
        target = Image.fromarray(img) if isinstance(img, np.ndarray) else img
        output = ImageOps.autocontrast(target, cutoff=self.cutoff, ignore=self.invalid)
        output = np.array(output) if isinstance(img, np.ndarray) else output
        return output