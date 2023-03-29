from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from einops import rearrange

from ..geoutils.raster import (get_raster_geospatial, get_raster_info, load_image, save_image)
from ..utils.search import file_search


class Slicer:

    def __init__(self,
                 filelist: str = 'test.txt',
                 directory: str = "data",
                 width: int = 1024,
                 height: int = 1024,
                 horizontal_overlap: float = 1.,
                 vertical_overlap: float = 1.,
                 sr_use: bool = True) -> None:

        with open(filelist, mode='r') as f:
            extension = file_search('image', directory)[0].split('.')[-1]
            filenames = [f"{s.strip()}.{extension}" for s in f.readlines()]

        params = []
        horizontal_overlap = 1. if horizontal_overlap <= 0. else horizontal_overlap
        vertical_overlap = 1. if vertical_overlap <= 0. else vertical_overlap
        self.reference = directory

        for filename in filenames:
            raster_info = get_raster_info(f"{directory}/{filename}")
            x_list = range(0, raster_info.width, int(width / horizontal_overlap))
            y_list = range(0, raster_info.height, int(height / vertical_overlap))
            params.extend([[filename, x, y, width, height] for y, x in product(y_list, x_list)])
        self.slice_params = pd.DataFrame(params, columns=['filename', 'x', 'y', 'w', 'h'])

    def cut(self, idx: int, directory: str, scale: float = 1.) -> np.ndarray:
        slice_param = self.slice_params.iloc[idx]
        image = load_image(f"{directory}/{slice_param.filename}",
                           x=int(slice_param.x * scale),
                           y=int(slice_param.y * scale),
                           w=int(slice_param.w * scale),
                           h=int(slice_param.h * scale))
        return rearrange(image, "c h w -> h w c") if len(image) > 2 else image

    def save(self, patches: torch.Tensor, output_directory: str, scale: float = 1.):
        for filename in self.slice_params.filename.unique():
            df = self.slice_params.query(f"filename == \'{filename}\'") if len(
                self.slice_params.filename.unique()) > 1 else self.slice_params
            merged_image = self.merge(patches, filename, df, scale=scale)

            ginfo = get_raster_geospatial(f"{self.reference}/{filename}")
            if not ginfo.transform == (0., 1., 0., 1., 0., 1.):
                ginfo.transform[1] /= scale
                ginfo.transform[5] /= scale

            kwargs = dict(filename=f"{output_directory}/{Path(filename).stem}.tif",
                          image=merged_image,
                          transform=tuple(ginfo.transform) if not ginfo.transform == (0., 1., 0., 1., 0., 1.) else None,
                          projection=ginfo.projection)
            save_image(**kwargs)

    def merge(self, patches: torch.Tensor, filename: str, df: pd.DataFrame, scale: float = 1.) -> np.ndarray:
        raster_info = get_raster_info(f"{self.reference}/{filename}")
        raster_info.height = int(raster_info.height * scale)
        raster_info.width = int(raster_info.width * scale)
        band = 1 if len(patches.shape) == 3 else patches.shape[1]
        output = np.zeros((band, raster_info.height, raster_info.width), dtype=np.float32)
        count = np.zeros((band, raster_info.height, raster_info.width), dtype=np.float32)

        for row in df.itertuples():
            start_x = int(row.x * scale)
            start_y = int(row.y * scale)
            width = int(row.w * scale)
            height = int(row.h * scale)
            slice_x = slice(start_x, start_x + width if start_x + width < raster_info.width else None)
            slice_y = slice(start_y, start_y + height if start_y + height < raster_info.height else None)
            patch = patches[row.Index]
            patch = patch[np.newaxis, :, :] if len(patch.shape) == 2 else patch

            w = width if start_x + width < raster_info.width else raster_info.width - start_x
            h = height if start_y + height < raster_info.height else raster_info.height - start_y
            output[:, slice_y, slice_x] += patch[:, :h, :w]
            count[:, slice_y, slice_x] += np.ones_like(patch[:, :h, :w], dtype=np.float32)
        count = np.where(count == 0., 1., count)
        return (output / count).squeeze()

    def __len__(self) -> int:
        return len(self.slice_params)
