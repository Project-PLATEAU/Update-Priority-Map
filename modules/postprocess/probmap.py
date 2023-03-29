import concurrent.futures as cf
import os

import geopandas as gpd
import pandas as pd
from rich.progress import track

from ..geoutils.vector import polygonize


def polygonize_probmaps(directory: str, filelist: str, threshold: float = 0.5) -> gpd.GeoDataFrame:
    with open(filelist, mode='r') as f:
        filenames = [f"{s.strip()}.tif" for s in f.readlines()]

    with cf.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(polygonize, **dict(threshold=threshold, filename=f"{directory}/{f}", item=dict(file=f)))
            for f in filenames
        ]

        output = gpd.GeoDataFrame()
        for future in track(futures, description="Polygonize probmaps..."):
            output = pd.concat([output, future.result()]).reset_index(drop=True)
    return output
