"""Small JSON configurations with explicit, relative inheritance."""
from copy import deepcopy
import json
from pathlib import Path


def merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path, _seen=None):
    path = Path(path).resolve()
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        raise ValueError("Circular config inheritance")
    seen.add(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    parent = value.pop("extends", None)
    if parent:
        return merge(load_config(path.parent / parent, seen), value)
    return value
