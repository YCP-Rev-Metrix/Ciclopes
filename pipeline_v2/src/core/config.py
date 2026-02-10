from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List
from pathlib import Path
import yaml
import re


def deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def interpolate_variables(cfg: dict) -> dict:
    pattern = r'\$\{([^}]+)\}'
    full_pattern = re.compile(r'^\$\{([^}]+)\}$')

    def resolve_value(value, cfg_root):
        if isinstance(value, str):
            full_match = full_pattern.match(value)
            if full_match:
                path = full_match.group(1)
                keys = path.split('.')
                result = cfg_root
                try:
                    for key in keys:
                        result = result[key]
                    return result
                except (KeyError, TypeError):
                    return value

            def replace_var(match):
                path = match.group(1)
                keys = path.split('.')
                result = cfg_root
                try:
                    for key in keys:
                        result = result[key]
                    return str(result)
                except (KeyError, TypeError):
                    return match.group(0)

            return re.sub(pattern, replace_var, value)
        elif isinstance(value, dict):
            return {k: resolve_value(v, cfg_root) for k, v in value.items()}
        elif isinstance(value, list):
            return [resolve_value(item, cfg_root) for item in value]
        else:
            return value

    return resolve_value(cfg, cfg)


@dataclass
class Config:
    run: Dict[str, Any] = field(default_factory=dict)
    trainer: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    wandb: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path, base_configs: List[str | Path] | None = None):
        merged: Dict[str, Any] = {}

        if base_configs:
            for base_path in base_configs:
                if Path(base_path).exists():
                    with open(base_path, "r", encoding="utf-8") as f:
                        base = yaml.safe_load(f) or {}
                    merged = deep_merge(merged, base)

        with open(path, "r", encoding="utf-8") as f:
            main = yaml.safe_load(f) or {}

        merged = deep_merge(merged, main)
        merged = interpolate_variables(merged)

        allowed = ("run", "trainer", "model", "data", "training", "wandb")
        filtered = {k: merged.get(k, {}) for k in allowed}

        return cls(**filtered)

    @classmethod
    def from_experiment(cls, exp_name: str):
        project_root = Path(__file__).resolve().parents[2]
        common_path = project_root / "configs/base/common.yaml"
        exp_path = project_root / "configs/exp" / f"{exp_name}.yaml"

        base_configs = [common_path]
        return cls.load(exp_path, base_configs=base_configs)
