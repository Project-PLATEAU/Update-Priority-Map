from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from rich.progress import track

from ..utils.search import file_search
from .raster import create_vrt, get_raster_resolution
from .vector import get_footprint


def get_neighbors(row: gpd.GeoSeries, reference: gpd.GeoDataFrame, tolerance=3.) -> list[str]:
    resolutions = get_raster_resolution(row.filename)
    resolution = min(resolutions.x, abs(resolutions.y))
    return [
        n
        for n in reference[~reference.geometry.disjoint(row.geometry.buffer(resolution * tolerance))].filename.tolist()
        if not n == row.filename
    ]


def create_vrt_image(src_path: str, dst_path: str) -> pd.DataFrame:
    Path(dst_path).mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame([[f, get_footprint(f)] for f in file_search('image', src_path, add_dir=True)],
                           columns=['filename', 'geometry'])
    gdf['neighbors'] = gdf.apply(get_neighbors, reference=gdf, axis=1)

    for filename, neighbor in track(zip(gdf.filename.values, gdf.neighbors.values), description="Create VRT image"):
        create_vrt([filename] + neighbor, f"{dst_path}/{Path(filename).with_suffix('.vrt').name}")
    return gdf.drop(['geometry'], axis=1)
