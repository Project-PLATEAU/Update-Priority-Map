import re

import albumentations as alb
import numpy as np


class NormalizeCD:

    def __init__(self, mean: list[float], std: list[float], max_pixel_value: float, **kwargs) -> None:
        self.max_pixel_value = max_pixel_value
        self.normalize = alb.Normalize(mean=mean, std=std, max_pixel_value=max_pixel_value, p=1., always_apply=True)

    def __call__(self, **kwargs: dict) -> dict:
        kwargs['image'] = self.normalize(**dict(image=kwargs['image']))['image']
        for key in kwargs.keys():
            if re.match(r'image[0-9]+', key):
                kwargs[key] = kwargs[key].astype(np.float32) / self.max_pixel_value
        return kwargs

    def __repr__(self) -> str:
        return f"Normalize(normalize={self.normalize})"
