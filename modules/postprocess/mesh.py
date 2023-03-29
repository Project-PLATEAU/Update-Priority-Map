from decimal import Decimal

import geopandas as gpd
import pandas as pd
from mapclassify import NaturalBreaks
from rich.progress import track
from shapely.geometry import Polygon

from ..geoutils.mesh import get_intersect_mesh
from ..geoutils.vector import get_footprint, filtered_small_polygon, get_clip_polygons
from ..utils.search import file_search

LATLON_PRECISION = Decimal(5e-14)  # 緯度経度座標の座標誤差
MIN_AREA = 10.  # 生成ポリゴンの最小面積[m^2]
CLIP_BUFFER = -10.  # メッシュデータのクリップ時の縮小バッファサイズ


def calculate_rate(mesh: Polygon, changed: gpd.GeoDataFrame, bldg_lod0: gpd.GeoDataFrame) -> tuple[float, float]:
    # Clip building & check valid
    changed.geometry = changed.geometry.buffer(0)
    clip_changed = changed.clip(mesh)
    clip_changed['check'] = clip_changed.geometry.is_empty | clip_changed.geometry.isna()
    changed_area = sum(clip_changed.query("check == False").drop('check', axis=1).reset_index(drop=True).area)

    clip_bldg = bldg_lod0.clip(mesh)
    clip_bldg['check'] = clip_bldg.geometry.is_empty | clip_bldg.geometry.isna()
    lod0_area = sum(clip_bldg.query("check == False").drop('check', axis=1).reset_index(drop=True).area)

    # Calculate changed ratio
    mesh_ratio = changed_area / mesh.area
    lod0_ratio = changed_area / lod0_area if lod0_area > 0. else 10. if changed_area > 0. else changed_area
    return mesh_ratio, lod0_ratio


def create_mesh(changed_gdf: gpd.GeoDataFrame,
                bldg_path: str,
                probmap_path: str,
                mesh_order: int,
                epsg: int,
                bldg_threshold: float = 100.) -> gpd.GeoDataFrame:
    # Projection to plane rectangular coordinate.
    changed_gdf = filtered_small_polygon(changed_gdf.to_crs(epsg=epsg), bldg_threshold)
    bldg_lod0_gdf = filtered_small_polygon(gpd.read_file(bldg_path).to_crs(epsg=epsg), bldg_threshold)
    footprints = [[get_footprint(f)] for f in file_search('image', probmap_path, add_dir=True)]
    aoi_gdf = gpd.GeoDataFrame(footprints, columns=['geometry']).buffer(0.).set_crs(epsg=6668)

    # Get mesh
    mesh = get_intersect_mesh(aoi_gdf.unary_union.buffer(LATLON_PRECISION), mesh_order).to_crs(epsg=epsg)
    mesh['check'] = mesh.geometry.apply(lambda x: x.area > MIN_AREA)
    mesh = mesh.query("check == True").drop('check', axis=1).reset_index(drop=True)

    mesh_code = get_clip_polygons(mesh, aoi_gdf.to_crs(epsg=epsg).buffer(CLIP_BUFFER).unary_union).code.to_list()
    mesh['check'] = mesh.code.apply(lambda x: x in mesh_code)
    mesh = mesh.query("check == True").drop('check', axis=1).reset_index(drop=True)

    output = []
    for m in track(mesh.geometry, description="Create mesh..."):
        mesh_ratio, lod0_ratio = calculate_rate(m, changed_gdf, bldg_lod0_gdf)
        output.append([mesh_ratio, lod0_ratio, m])
    return gpd.GeoDataFrame(output, columns=['mesh', 'lod0', 'geometry']).set_crs(epsg=epsg)


def classify_mesh(gdf: gpd.GeoDataFrame, threshold_file: str, n_classes: int = 5) -> gpd.GeoDataFrame:
    if not threshold_file == '':
        mesh = pd.read_csv(f"{threshold_file}_mesh.csv").threshold
        if gdf.mesh.max() > mesh.iloc[-1]:
            mesh.iloc[-1] = gdf.mesh.max()
        lod0 = pd.read_csv(f"{threshold_file}_lod0.csv").threshold
        if gdf.lod0.max() > lod0.iloc[-1]:
            lod0.iloc[-1] = gdf.lod0.max()
    else:
        n_classes = len(gdf) if len(gdf) < n_classes else n_classes
        mesh = NaturalBreaks(gdf.mesh, k=n_classes, initial=50).bins
        lod0 = NaturalBreaks(gdf.lod0, k=n_classes, initial=50).bins
    thresholds = pd.DataFrame([[-1, -1]] + [[m, l] for m, l in zip(mesh, lod0)], columns=['mesh', 'lod0'])

    gdf['label_mesh'] = pd.cut(gdf.mesh,
                               bins=thresholds.mesh.values.tolist(),
                               labels=[i for i in range(len(thresholds) - 1)],
                               include_lowest=True).fillna(0).astype(int)
    gdf['label_lod0'] = pd.cut(gdf.lod0,
                               bins=thresholds.lod0.values.tolist(),
                               labels=[i for i in range(len(thresholds) - 1)],
                               include_lowest=True).fillna(0).astype(int)
    return gdf
