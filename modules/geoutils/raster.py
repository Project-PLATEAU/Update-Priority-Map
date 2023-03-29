from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from osgeo import gdal, gdal_array
from rasterio.mask import mask
from rasterio.transform import Affine
from shapely.geometry import Polygon

from .vector import (get_clip_polygons, get_footprint, get_rasterio_transform_from_gdal, pixel_to_proj)


def load_image(filename: str,
               x: int = 0,
               y: int = 0,
               w: int = -1,
               h: int = -1,
               padding: str = 'constant') -> np.ndarray:
    g = gdal.Open(filename)
    w = g.RasterXSize if w < 1 else w
    h = g.RasterYSize if h < 1 else h

    # padding
    if (x + w) > g.RasterXSize or (y + h) > g.RasterYSize:
        pw = min(g.RasterXSize - x, w)
        ph = min(g.RasterYSize - y, h)
        pad_shape = [(0, 0), (0, h - ph), (0, w - pw)] if g.RasterCount > 1 else [(0, h - ph), (0, w - pw)]
        image = np.pad(g.ReadAsArray(xoff=x, yoff=y, xsize=pw, ysize=ph), pad_shape, padding)
    else:
        image = g.ReadAsArray(xoff=x, yoff=y, xsize=w, ysize=h)
    return image


def save_image(filename: str,
               image: np.ndarray,
               transform: Optional[Union[Affine, tuple[float]]] = None,
               projection: str = '',
               crs: str = 'EPSG:4326',
               driver: str = 'GTiff') -> None:
    Path(filename).resolve().parent.mkdir(parents=True, exist_ok=True)
    if isinstance(transform, Affine):
        save_image_with_rasterio(filename, image, transform, crs=crs, driver=driver)
    elif isinstance(transform, tuple):
        save_image_with_gdal(filename, image, transform, projection, driver=driver)
    else:
        cv_image = image if len(image.shape) == 2 else cv2.cvtColor(rearrange(image, "c h w -> h w c"),
                                                                    cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, cv_image)


def clip_image(filename: str, polygon: Polygon) -> tuple[np.ndarray, Affine]:
    with rasterio.open(filename) as src:
        patch, transform = mask(src, [polygon], crop=True, all_touched=True)
        return patch, transform


def rasterize(shape: Union[str, gpd.GeoDataFrame],
              reference_filename: str = '',
              image_info: DictConfig = None,
              transform: Affine = None,
              image_type: int = np.uint8,
              value: int = 1) -> tuple[np.ndarray, Affine]:
    if reference_filename == '':
        return rasterize_core(shape, image_info, transform, image_type=image_type, value=value)
    else:
        if isinstance(shape, str):
            return rasterize_from_file(shape, reference_filename, image_type=image_type, value=value)
        else:
            return rasterize_from_shape(shape, reference_filename, image_type=image_type, value=value)


def create_vrt(filenames: list[str], output_path: str, options: Optional[dict[str, Any]] = None) -> None:
    if options is not None:
        vrt_options = gdal.BuildVRTOptions(**options)
    else:
        vrt_options = gdal.BuildVRTOptions(resampleAlg='lanczos')
    gdal.BuildVRT(output_path, filenames, options=vrt_options)


def get_raster_info(data: Union[str, np.ndarray]) -> DictConfig:
    return get_raster_info_from_file(data) if isinstance(data, str) else get_raster_info_from_image(data)


def get_raster_stats(data: Union[str, np.ndarray], remove_invalid: bool = False, invalid_value: int = 0) -> DictConfig:
    if isinstance(data, str):
        result = get_raster_stats_from_file(data, remove_invalid=remove_invalid, invalid_value=invalid_value)
    else:
        result = get_raster_stats_from_image(data, remove_invalid=remove_invalid, invalid_value=invalid_value)
    return result


def get_raster_geospatial(filename: str, x: int = 0, y: int = 0) -> DictConfig:
    g = gdal.Open(filename)
    transform = list(g.GetGeoTransform())
    if x > 0 or y > 0:
        x, y = pixel_to_proj(x, y, transform)
        transform[0] = x
        transform[3] = y
    return OmegaConf.create(dict(transform=tuple(transform), projection=g.GetProjection()))


def get_raster_resolution(filename: str) -> DictConfig:
    transform = get_raster_geospatial(filename).transform
    return OmegaConf.create(dict(x=transform[1], y=transform[5]))


def save_image_with_gdal(filename: str,
                         image: np.ndarray,
                         transform: tuple[float],
                         projection: str,
                         driver: str = 'GTiff') -> None:
    output = gdal_array.OpenArray(image)
    output.SetGeoTransform(transform)
    if not projection == '':
        output.SetProjection(projection)
    driver = gdal.GetDriverByName(driver)
    driver.CreateCopy(filename, output)


def save_image_with_rasterio(filename: str,
                             image: np.ndarray,
                             transform: Affine,
                             crs: str = 'EPSG:4326',
                             driver: str = 'GTiff') -> None:
    image = image[np.newaxis, :, :] if len(image.shape) == 2 else image
    profile = dict(driver=driver,
                   height=image.shape[1],
                   width=image.shape[2],
                   count=image.shape[0],
                   dtype=image.dtype,
                   crs=crs,
                   transform=transform)
    with rasterio.open(filename, 'w', **profile) as dst:
        dst.write(image)


def rasterize_core(gdf: gpd.GeoDataFrame,
                   image_info: DictConfig,
                   transform: Affine,
                   image_type: int = np.uint8,
                   value: int = 1) -> tuple[np.ndarray, Affine]:
    if len(gdf) > 0:
        image = rasterio.features.rasterize(shapes=gdf.geometry,
                                            out_shape=(image_info.height, image_info.width),
                                            fill=0,
                                            transform=transform,
                                            all_touched=True,
                                            default_value=value,
                                            dtype=image_type)
    else:
        image = np.zeros((image_info.height, image_info.width), dtype=image_type)
    return image, transform


def rasterize_from_file(shape_filename: str,
                        image_filename: str,
                        image_type: int = np.uint8,
                        value: int = 1) -> tuple[np.ndarray, Affine]:
    gdf = gpd.read_file(shape_filename, mask=get_footprint(image_filename))
    image_info = get_raster_info(image_filename)
    transform = get_rasterio_transform_from_gdal(get_raster_geospatial(image_filename).transform)
    return rasterize_core(gdf, image_info, transform, image_type=image_type, value=value)


def rasterize_from_shape(gdf: gpd.GeoDataFrame,
                         image_filename: str,
                         image_type: int = np.uint8,
                         value: int = 1) -> tuple[np.ndarray, Affine]:
    gdf = get_clip_polygons(gdf, get_footprint(image_filename))
    image_info = get_raster_info(image_filename)
    transform = get_rasterio_transform_from_gdal(get_raster_geospatial(image_filename).transform)
    return rasterize_core(gdf, image_info, transform, image_type=image_type, value=value)


def get_raster_info_from_image(data: np.ndarray) -> DictConfig:
    data = data[np.newaxis, :, :] if len(data.shape) == 2 else data
    return OmegaConf.create(dict(band=data.shape[0], width=data.shape[2], height=data.shape[1], dtype=str(data.dtype)))


def get_raster_info_from_file(filename: str) -> DictConfig:
    g = gdal.Open(filename)
    dtype = str(g.ReadAsArray(xoff=0, yoff=0, xsize=1, ysize=1).dtype)
    return OmegaConf.create(dict(band=g.RasterCount, width=g.RasterXSize, height=g.RasterYSize, dtype=dtype))


def get_raster_stats_from_image(data: np.ndarray, remove_invalid: bool = False, invalid_value: int = 0) -> DictConfig:
    output = dict()
    data = np.where(data == invalid_value, np.nan, data) if remove_invalid else data
    minimum = np.nanmin(data, axis=(1, 2)).reshape(data.shape[0], 1, 1).tolist()
    maximum = np.nanmax(data, axis=(1, 2)).reshape(data.shape[0], 1, 1).tolist()
    mean = np.nansum(data, axis=(1, 2)).reshape(data.shape[0], 1, 1).tolist()
    stddev = np.nanstd(data, axis=(1, 2)).reshape(data.shape[0], 1, 1).tolist()

    for band in range(data.shape[0]):
        stat_dict = dict(minimum=minimum[band], maximum=maximum[band], mean=mean[band], stddev=stddev[band])
        output[band] = stat_dict
    return OmegaConf.create(output)


def get_raster_stats_from_file(filename: str, remove_invalid: bool = False, invalid_value: int = 0) -> DictConfig:
    data = gdal.Open(filename)
    if remove_invalid:
        return get_raster_stats_from_image(data.ReadAsArray(),
                                           remove_invalid=remove_invalid,
                                           invalid_value=invalid_value)
    output = dict()
    for band in range(data.RasterCount):
        stats = data.GetRasterBand(band + 1).GetStatistics(0, 1)
        stat_dict = dict(minimum=stats[0], maximum=stats[1], mean=stats[2], stddev=stats[3])
        output[band] = stat_dict
    return OmegaConf.create(output)
