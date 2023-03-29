import argparse
import os
from glob import glob

import geopandas as gpd
from rich.progress import track

from modules.loggers.logger import StandardLogger
from modules.utils.config import get_config
from modules.utils.timer import Timer


def styling_map(src_gdf: gpd.GeoDataFrame, style_cfg: dict) -> gpd.GeoDataFrame:
    num_of_mesh = len(src_gdf)
    # Stroke setting
    stroke_list = [style_cfg['stroke']] * num_of_mesh
    # Width of stroke setting
    stroke_width_list = [style_cfg['stroke-width']] * num_of_mesh
    # Opacity of stroke setting
    stroke_opacity_list = [style_cfg['stroke-opacity']] * num_of_mesh
    # Fill setting
    column_name = 'label_' + style_cfg['denom']
    fill_cfg_list = style_cfg['fill']
    fill_list = [fill_cfg_list[r] for r in src_gdf[column_name]]

    # Priority level setting
    priority_list = [level + 1 for level in src_gdf[column_name]]

    # Opacity of fill setting
    fill_opacity_list = [style_cfg['fill-opacity']] * num_of_mesh

    styled_gdf = gpd.GeoDataFrame([])
    styled_gdf['更新優先度'] = priority_list
    styled_gdf['stroke'] = stroke_list
    styled_gdf['stroke-width'] = stroke_width_list
    styled_gdf['stroke-opacity'] = stroke_opacity_list
    styled_gdf['fill'] = fill_list
    styled_gdf['fill-opacity'] = fill_opacity_list
    styled_gdf['geometry'] = src_gdf['geometry']

    return styled_gdf


def main():
    parser = argparse.ArgumentParser(description='Script to style update priority map.')
    parser.add_argument('--config', type=str, default='./conf/config.yml', help='Path of config file.')
    args = parser.parse_args()

    style_cfg = get_config('style', args.config)
    logger = StandardLogger(log_filename="log/style_mesh.log", stdout=True)
    logger.configure('style_mesh')
    timer = Timer(device='cpu')

    # Style Processing
    mesh_file_path_list = glob(os.path.join(style_cfg['mesh'], '*.geojson'))
    for mesh_file_path in track(mesh_file_path_list, description='Style mesh...'):
        src_gdf = gpd.read_file(mesh_file_path)
        styled_gdf = styling_map(src_gdf, style_cfg)
        styled_gdf.to_crs(epsg=4326, inplace=True)

        # Output style result
        save_filename = os.path.basename(mesh_file_path).split('.')[0] + '_' + style_cfg['denom'] + '.geojson'
        save_json_path = os.path.join(style_cfg['output_dir'], save_filename)
        styled_gdf.to_file(save_json_path, driver='GeoJSON')
        logger.log('info', timer.print(f"Style {os.path.basename(mesh_file_path)}", mode='lap'))
    logger.log('info', timer.print("Done priority mesh style.", mode='total'))


if __name__ == "__main__":
    main()
