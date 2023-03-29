from __future__ import annotations

import re
from pathlib import Path

PATTERNS = dict(file=r".*.(csv|tsv|txt|pkl|tfw|jfw|jgw|pgw)$",
                image=r".*.(jpg|jpeg|png|tif|tiff|jp2|j2k|jpf|jpx|jpm|mj2|vrt)$",
                shape=r".*.(shp|geojson|gpkg|parquet)$",
                config=r".*.(yml|yaml|json)$",
                code=r".*.(sh|py)$",
                checkpoint=r".*.(ckpt|pth|tar)$")


def file_search(target: str, directory: str, patterns: list[str] = [], add_dir: bool = False) -> list[str]:
    directory = Path(directory)
    if target == 'directory':
        directories = sorted([d for d in directory.iterdir() if d.is_dir()])
        return sorted([str(d) for d in directories]) if add_dir else sorted([d.name for d in directories])
    else:
        patterns = PATTERNS[target] if target in PATTERNS.keys() else patterns
        filenames = sorted([f for f in directory.iterdir() if re.search(patterns, str(f), re.IGNORECASE)])
        return sorted([str(f) for f in filenames]) if add_dir else sorted([f.name for f in filenames])
