import argparse

from modules.configures.configure import get_trainer
from modules.inference.config import set_config
from modules.utils.config import get_config


def main(args: argparse.Namespace) -> None:
    cfg = set_config(get_config('generate_probmap', args.config))
    trainer = get_trainer(**dict(name=cfg.configure.name, config=cfg))
    trainer.predict()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Generate probability map.")
    parser.add_argument('--config', type=str, default="conf/config.yml", help="Path to config yaml file.")
    args, _ = parser.parse_known_args()
    main(args)
