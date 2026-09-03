"""Tiny YAML config loader with attribute access and dict merging."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Mapping, Union

import yaml


class Config(dict):
    """``dict`` subclass allowing ``cfg.key`` in addition to ``cfg["key"]``.

    Nested mappings are wrapped recursively on access.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[name] = value
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if isinstance(node, Mapping) and part in node:
                node = node[part]
            else:
                return default
        return node


def _merge(base: Dict, override: Mapping) -> Dict:
    out = copy.deepcopy(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, Mapping):
            out[key] = _merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_config(path: Union[str, Path], **overrides: Any) -> Config:
    """Load a YAML file into a :class:`Config`, applying keyword overrides on top."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    if overrides:
        data = _merge(data, overrides)
    return Config(data)


def dump_config(cfg: Mapping, path: Union[str, Path]) -> None:
    """Serialise a resolved config next to a run's outputs."""
    plain = _to_plain(cfg)
    Path(path).write_text(yaml.safe_dump(plain, sort_keys=False))


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj
