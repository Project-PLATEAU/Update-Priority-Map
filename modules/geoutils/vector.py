from __future__ import annotations

from typing import Any, Optional, Union

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from omegaconf import DictConfig, OmegaConf
from osgeo import gdal
from rasterio.features import shapes
from rasterio.transform import Affine
from shapely.geometry import Polygon, MultiPolygon


def get_corner(filename: str) -> DictConfig:
    g = gdal.Open(filename)
    transform = g.GetGeoTransform()
    ulx = transform[0]
    uly = transform[3]
    lrx = ulx + g.RasterXSize * transform[1]
    lry = uly + g.RasterYSize * transform[5]
    return OmegaConf.create(dict(ulx=ulx, uly=uly, lrx=lrx, lry=lry))


def get_footprint(filename: str) -> Polygon:
    corner = get_corner(filename)
    return Polygon([(corner.ulx, corner.uly), (corner.lrx, corner.uly), (corner.lrx, corner.lry),
                    (corner.ulx, corner.lry)])


def get_crs(filename: str) -> str:
    with rasterio.open(filename) as src:
        return str(src.crs)


def get_rasterio_transform_from_gdal(gdal_transform: tuple[float]) -> Affine:
    return Affine.from_gdal(*gdal_transform)


def get_intersect_polygons(polygon: gpd.GeoDataFrame, aoi: Polygon) -> gpd.GeoDataFrame:
    polygon['check'] = polygon.geometry.apply(lambda x: aoi.intersects(x))
    return polygon.query("check == True").drop('check', axis=1)


def get_clip_polygons(polygon: gpd.GeoDataFrame, aoi: Polygon) -> gpd.GeoDataFrame:
    polygon = get_intersect_polygons(polygon, aoi)
    polygon.geometry = polygon.geometry.intersection(aoi)
    polygon['check'] = polygon.geometry.apply(lambda x: isinstance(x, (Polygon, MultiPolygon)) and not x.is_empty)
    return polygon.query("check == True").drop('check', axis=1).reset_index(drop=True)


def filtered_small_polygon(polygon: gpd.GeoDataFrame, threshold: float) -> gpd.GeoDataFrame:
    polygon['check'] = polygon.area > threshold
    return polygon.query("check == True").drop('check', axis=1).reset_index(drop=True)


def proj_to_pixel(x: float, y: float, transform: tuple[float]) -> tuple(int):
    px = int((x - transform[0] - y * transform[2]) / transform[1])
    py = int((y - transform[3] - x * transform[4]) / transform[5])
    return px, py


def pixel_to_proj(px: Union[int, float], py: Union[int, float], transform: tuple[float]) -> tuple[float]:
    x = transform[0] + px * transform[1] + py * transform[2]
    y = transform[3] + px * transform[4] + py * transform[5]
    return x, y


def polygonize(threshold: Union[int, float],
               filename: str = '',
               image: Optional[np.ndarray] = None,
               transform: Optional[tuple[float]] = None,
               connectivity: int = 4,
               item: dict[str, Any] = {}) -> gpd.GeoDataFrame:
    if image is None:
        return polygonize_from_file(filename, threshold, connectivity=connectivity, item=item)
    else:
        return polygonize_from_image(image,
                                     get_rasterio_transform_from_gdal(transform),
                                     threshold,
                                     connectivity=connectivity,
                                     item=item)


def polygonize_from_file(filename: str,
                         threshold: Union[int, float],
                         connectivity: int = 4,
                         item: dict[str, Any] = {}) -> gpd.GeoDataFrame:
    with rasterio.open(filename) as src:
        if len(src.indexes) == 1:  # Single band
            image = src.read()
        elif len(src.indexes) == 3:  # RGB band
            image = cv2.cvtColor(src.read().transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
        else:
            image = np.sum(src.read(), axis=0)

        gdf = polygonize_from_image(image, src.transform, threshold, connectivity=connectivity, item=item)
        if len(gdf) > 0 and src.crs is not None:
            gdf = gdf.set_crs(crs=src.crs)
        return gdf


def polygonize_from_image(image: np.ndarray,
                          transform: Affine,
                          threshold: Union[int, float],
                          connectivity: int = 4,
                          item: dict[str, Any] = {}) -> gpd.GeoDataFrame:
    binarize = np.where(image > threshold, 1, 0).squeeze().astype(np.uint8)

    # Filtering polygon using number of valid pixel
    contours, _ = cv2.findContours(binarize, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) > 0.]
    binarize = cv2.drawContours(binarize.copy(), contours, -1, 1, -1)

    polygon = ({
        'properties': item,
        'geometry': s
    } for s, _ in shapes(binarize, mask=binarize, connectivity=connectivity, transform=transform))
    return gpd.GeoDataFrame.from_features(list(polygon))
