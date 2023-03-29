from __future__ import annotations

import re
from typing import Any, Union

import albumentations as al
import numpy as np
import torch
import torchvision.transforms as tf
from albumentations.core.transforms_interface import BasicTransform
from einops import rearrange
from PIL import Image


class SelfCompose:

    def __init__(self, transforms: list[Any] = []):
        self.transforms = transforms

    def __call__(self, **kwargs: dict) -> dict:
        for t in self.transforms:
            kwargs = t(**kwargs)
        return kwargs

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class IntegrateCompose:

    def __init__(self, transforms: list[Any] = []):
        self.set_compose(transforms)

    def __call__(self, **kwargs) -> dict:
        for t in self.transforms:
            if isinstance(t, tf.Compose):
                kwargs['image'] = t(self.convert_image_type(kwargs, ['image'], torch.Tensor)['image'])
            else:
                kwargs = t(**self.convert_image_type(kwargs, ['image', 'mask'], np.ndarray))
        return self.convert_image_type(kwargs, ['image', 'mask'], torch.Tensor)

    def convert_image_type(self, args: dict, keys: list[str], image_type: Union[Image.Image, np.ndarray,
                                                                                torch.Tensor]) -> dict:
        for arg_key, value in args.items():
            if any([re.match(key, arg_key) for key in keys]) and not isinstance(value, image_type):
                if image_type == torch.Tensor:
                    args[arg_key] = self.convert_tensor(value)
                elif image_type == Image.Image:
                    args[arg_key] = self.convert_pil(value)
                else:
                    args[arg_key] = self.convert_ndarray(value)
        return args

    def set_compose(self, transforms: list[Any]) -> None:
        # Separate transforms by library
        if transforms == []:
            self.transforms = []
            return

        compose_keys = []
        for t in transforms:
            if isinstance(t, BasicTransform):
                compose_keys.append('albumentations')
            elif hasattr(tf, t.__class__.__name__):
                compose_keys.append('torchvision')
            else:
                compose_keys.append('self')

        # Set component
        composes = []
        start_key = ''
        start_idx = 0
        for i, key in enumerate(compose_keys):
            if start_key == '':
                start_key = key
                start_idx = i
            elif not start_key == key:
                composes.append(getattr(self, f"get_{start_key}_compose")(transforms[start_idx:i]))
                start_key = key
                start_idx = i
        composes.append(getattr(self, f"get_{start_key}_compose")(transforms[start_idx:]))
        self.transforms = composes

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '(\n'
        for t in self.transforms:
            if isinstance(t, al.Compose):
                format_string += 'Albumentations '
            elif isinstance(t, tf.Compose):
                format_string += 'Torchvision '
            else:
                format_string += 'Modules '
            format_string += f"{t.__repr__()}\n"
        return format_string + ')'

    @staticmethod
    def get_albumentations_compose(transforms: Union[BasicTransform, list[BasicTransform]]) -> al.Compose:
        return al.Compose(transforms, additional_targets=dict(image0='image', image1='image', mask0='mask'))

    @staticmethod
    def get_torchvision_compose(transforms: Union[Any, list[Any]]) -> tf.Compose:
        return tf.Compose(transforms)

    @staticmethod
    def get_self_compose(transforms: Union[Any, list[Any]]):
        return SelfCompose(transforms)

    @staticmethod
    def convert_tensor(image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        if isinstance(image, Image.Image):
            image = tf.PILToTensor(image)
        else:
            image = rearrange(torch.from_numpy(image), "h w c -> c h w") if image.ndim == 3 else torch.from_numpy(image)
        return image

    @staticmethod
    def convert_pil(image: Union[np.ndarray, torch.tensor]) -> Image.Image:
        return Image.fromarray(image) if isinstance(image, np.ndarray) else tf.ToPILImage(image)

    @staticmethod
    def convert_ndarray(image: Union[Image.Image, torch.Tensor]) -> np.ndarray:
        if isinstance(image, Image.Image):
            image = np.array(image)
        else:
            image = rearrange(image, "c h w -> h w c").numpy() if image.ndim == 3 else image.numpy()
        return image
