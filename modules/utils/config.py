from omegaconf import DictConfig, OmegaConf


def get_config(key: str, config_path: str) -> DictConfig:
    return OmegaConf.merge(OmegaConf.load(config_path)[key], OmegaConf.from_cli())
