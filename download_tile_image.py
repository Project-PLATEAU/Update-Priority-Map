import argparse
import asyncio
import math
import os
import shutil
import time
from io import BytesIO

os.environ['USE_PYGEOS'] = '0'
os.environ['NO_PROXY'] = 'localhost'
import geopandas as gpd
import httpx
import numpy as np
from osgeo import gdal
from PIL import Image
from rich.progress import track

from modules.loggers.logger import StandardLogger
from modules.utils.config import get_config
from modules.utils.timer import Timer


def get_map_tile_coord(lon: float, lat: float, zoom_level: int) -> tuple[int, int]:
    n = 2.0 ** zoom_level

    # Calculate X coordinate of tile
    x_tile = int((lon + 180.0) / 360.0 * n)

    # Calculate Y coordinate of tile
    lat_rad = math.radians(lat)
    y_tile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)

    return (x_tile, y_tile)


def get_map_tile_bbox(z: int, x: int, y: int) -> list[float, float, float, float]:
    def convert_tile_coord_to_latlon(x_tile, y_tile, zoom_level):
        n = 2.0 ** zoom_level
        lon = x_tile / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y_tile / n)))
        lat = math.degrees(lat_rad)
        return (lon, lat)
    # Geogoraphic coordinate of upper right
    ur_coord = convert_tile_coord_to_latlon(x + 1, y, z)
    # Geogoraphic coordinate of lower left
    ll_coord = convert_tile_coord_to_latlon(x, y + 1, z)

    return [ll_coord[0], ll_coord[1], ur_coord[0], ur_coord[1]]


async def dl_map_tile_image(zoom_level: int, x_tile: int, y_tile: int, base_url: str, client) -> np.array:
    url = '{}/{}/{}/{}.png'.format(base_url, zoom_level, x_tile, y_tile)

    blank = np.zeros((256, 256, 3), dtype=np.uint8)  # Tile for 0-fill

    # Download tile images as binary data
    response = await client.get(url)

    if response.status_code == 200:
        # Convert from binary data to Numpy array
        return np.array(Image.open(BytesIO(response.content)).convert('RGB'))
    else:
        return blank


async def get_concated_map_tile_image(zoom_level: int, tl_x_tile: int, tl_y_tile: int, base_url: str,
                                      num_x_tiles: int, num_y_tiles: int) -> np.ndarray:
    row_list = []

    for y in range(num_y_tiles):
        async with httpx.AsyncClient() as client:
            col_tasks = [dl_map_tile_image(zoom_level, tl_x_tile + x, tl_y_tile + y, base_url, client) for x in range(num_x_tiles)]
            column_list = await asyncio.gather(*col_tasks, return_exceptions=False)

        time.sleep(1)

        row_list.append(np.hstack(column_list))

    return np.vstack(row_list)


def save_image(concated_tile_array: np.array, save_dir: str, save_name: str, scene: str, output_bounds: list, clip_bounds: list):
    # Path to save setting
    tmp_save_path = os.path.join(save_dir, 'tmp', save_name + '.png')
    bounds_save_path = os.path.join(save_dir, 'tmp', save_name + '_bounds.png')
    save_path = os.path.join(save_dir, save_name + '.png')

    height, width = concated_tile_array.shape[:2]

    # Convert Numpy array to PIL image and save
    pil_image = Image.fromarray(concated_tile_array)
    pil_image.save(tmp_save_path)

    # Obtain location using `gdal_translate`
    bounds_dst = gdal.Translate(bounds_save_path, tmp_save_path, format='PNG', outputType=gdal.GDT_Byte,
                                resampleAlg='lanczos', outputBounds=output_bounds, outputSRS='EPSG:4326')
    bounds_dst = None

    # Clip 3rd mesh
    if scene == 'old':
        dst = gdal.Warp(save_path, bounds_save_path, format='PNG', outputBounds=clip_bounds,
                        outputBoundsSRS='EPSG:4326', resampleAlg='lanczos', width=3136, height=2092)
    elif scene == 'new':
        dst = gdal.Warp(save_path, bounds_save_path, format='PNG', outputBounds=clip_bounds,
                        outputBoundsSRS='EPSG:4326', resampleAlg='lanczos', width=1568, height=1046)
    dst = None


def main():
    parser = argparse.ArgumentParser(description='Script to download XYZ tiles.')
    parser.add_argument('scene', choices=['old', 'new'], type=str, help='Type `old` for aerial images or `new` for ALOS-3 images.')
    parser.add_argument('--config', default='./conf/config.yml', type=str, help='Path to config yaml file')
    args = parser.parse_args()

    data_cfg = None
    if args.scene == 'old':
        data_cfg = get_config('aerial_download', args.config)
    elif args.scene == 'new':
        data_cfg = get_config('alos3_download', args.config)

    logger = StandardLogger(log_filename=f"log/download_image_{args.scene}.log", stdout=True)
    logger.configure('download_image')
    timer = Timer(device='cpu')

    # Download URL setting
    base_url = data_cfg['url']
    aoi_txt_path = data_cfg['mesh_code']
    save_dir = data_cfg['output']
    save_dir = os.path.join(save_dir, args.scene)
    zoom_level = data_cfg['zoom_level']

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'tmp'), exist_ok=True)

    mesh_gdf = gpd.read_file('./data/Japan-3rd-mesh/Japan-3rd-mesh.shp')

    # Load a text file that written 3rd mesh codes.
    with open(aoi_txt_path, mode='r') as f:
        mesh_code_list = [text_line.strip() for text_line in f.readlines()]

    logger.log('info', timer.print("Load setting yaml & 3rd mesh data", mode='lap'))

    for mesh_code in track(mesh_code_list, description='Download image...'):
        geom_bounds = mesh_gdf[mesh_gdf['Name'] == mesh_code]['geometry'].bounds
        min_x, min_y, max_x, max_y = geom_bounds.values[0]

        '''Calculate the maximum and minimum tile coorinates
           from the geographic coorinate of the for courners.'''
        tile_min_x, tile_max_y = get_map_tile_coord(min_x, min_y, zoom_level)
        tile_max_x, tile_min_y = get_map_tile_coord(max_x, max_y, zoom_level)

        ulx, _, _, uly = get_map_tile_bbox(zoom_level, tile_min_x, tile_min_y)
        _, lry, lrx, _ = get_map_tile_bbox(zoom_level, tile_max_x, tile_max_y)

        num_x_tiles = (tile_max_x - tile_min_x) + 1
        num_y_tiles = (tile_max_y - tile_min_y) + 1

        # Downloaded and concat map tile image
        concated_tile_image_array = asyncio.run(get_concated_map_tile_image(zoom_level, tile_min_x, tile_min_y, base_url,
                                                                            num_x_tiles, num_y_tiles))

        save_image(concated_tile_image_array, save_dir, mesh_code, args.scene, output_bounds=[ulx, uly, lrx, lry],
                   clip_bounds=[min_x, min_y, max_x, max_y])

        logger.log('info', timer.print(f"Download {mesh_code}", mode='lap'))
        time.sleep(1)

    shutil.rmtree(os.path.join(save_dir, 'tmp'))
    logger.log('info', timer.print("Done download image.", mode='total'))


if __name__ == "__main__":
    main()
