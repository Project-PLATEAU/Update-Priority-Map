from __future__ import annotations

from functools import wraps
from importlib import import_module, util
from typing import Any, Optional, Union


# Check if target is in modules
def check_module(modules: list[str], target: str) -> Optional[Any]:
    if '.' in target:
        add_module = target.rsplit('.', 1)[0]
        target = target.rsplit('.', 1)[1]
        modules = [f"{m}.{add_module}" if util.find_spec(f"{m}.{add_module}") else m for m in modules]

    is_exist = [hasattr(import_module(module), target) for module in modules]
    return getattr(import_module(modules[is_exist.index(True)]), target) if sum(is_exist) else None


# Decorator of module calling
def call_module(modules: Union[str, list[str]]):
    modules = [modules] if isinstance(modules, str) else modules

    def call_func(func):

        @wraps(func)
        def loader(**kwargs):
            kwargs['instance'] = check_module(modules, kwargs['name'])
            if kwargs['instance'] is None:
                raise FileNotFoundError(f"{kwargs['name']} is not found.")
            return func(**kwargs)

        return loader

    return call_func
