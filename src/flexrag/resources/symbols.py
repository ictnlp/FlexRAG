from __future__ import annotations

import importlib
from typing import Any


def symbol_path(obj: object) -> str:
    """Return an importable ``module:qualname`` path for an object."""
    return f"{obj.__module__}:{obj.__qualname__}"


def resolve_symbol(path: str) -> Any:
    """Resolve an object from a ``module:qualname`` path."""
    module_name, qualname = path.split(":", maxsplit=1)
    obj = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj
