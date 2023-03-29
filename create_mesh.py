import argparse
import warnings
from pathlib import Path

from modules.postprocess import probmap, mesh
from modules.loggers.logger import StandardLogger
from modules.utils.timer import Timer
from modules.utils.config import get_config

warnings.simplefilter('ignore', FutureWarning)


def main(args: argparse.Namespace):
    cfg = get_config('create_mesh', args.config)
    logger = StandardLogger(log_filename="log/create_mesh.log", stdout=True)
    logger.configure('create_mesh')
    timer = Timer(device='cpu')
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)

    changed_gdf = probmap.polygonize_probmaps(cfg.probmap, cfg.target)
    logger.log('info', timer.print("Polygonize probmaps", mode='lap'))

    for order in [3, 4, 5]:
        changed_ratio_gdf = mesh.create_mesh(changed_gdf, cfg.bldg, cfg.probmap, order, cfg.epsg)
        if order == 3:
            logger.log('info', timer.print("Create 3rd mesh", mode='lap'))
            threshold_file = f"{cfg.threshold}_3rd" if not cfg.threshold == '' else ''
            output_file = f"{cfg.output}/{cfg.filename}_3rd.geojson"
        else:
            logger.log('info', timer.print(f"Create {order}th mesh", mode='lap'))
            threshold_file = f"{cfg.threshold}_{order}th" if not cfg.threshold == '' else ''
            output_file = f"{cfg.output}/{cfg.filename}_{order}th.geojson"

        output_gdf = mesh.classify_mesh(changed_ratio_gdf, threshold_file)
        output_gdf.to_crs(epsg=6668).to_file(output_file)
    logger.log('info', timer.print("Done priority mesh create.", mode='total'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create priority mesh.")
    parser.add_argument('--config', type=str, default="conf/config.yml", help="Path to config yaml file.")
    args, _ = parser.parse_known_args()
    main(args)
