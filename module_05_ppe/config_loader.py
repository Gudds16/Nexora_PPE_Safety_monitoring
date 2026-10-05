"""Load and validate config/config.yaml and resolve paths."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

logger = logging.getLogger(__name__)

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
DEFAULT_CONFIG = MODULE_DIR / "config" / "config.yaml"

REQUIRED_SECTIONS = ("module", "classes", "model", "tracking", "association",
                     "event", "severity", "evidence", "output", "ppe_requirements")


class ConfigError(ValueError):
    """Raised when the configuration is missing or invalid."""


def resolve_path(path: Union[str, Path]) -> Path:
    """Resolve a config path: absolute stays, relative is taken from module_05_ppe/."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else (MODULE_DIR / p)


def find_config(config_path: Optional[Union[str, Path]]) -> Path:
    """Find the config file: as given (cwd), then inside module_05_ppe/, then default."""
    if config_path is None:
        return DEFAULT_CONFIG
    candidate = Path(config_path).expanduser()
    if candidate.is_file():
        return candidate
    inside_module = MODULE_DIR / candidate
    if inside_module.is_file():
        return inside_module
    raise ConfigError(f"Config file not found: {config_path} (also tried {inside_module})")


def load_config(config_path: Optional[Union[str, Path]] = "config/config.yaml") -> Dict[str, Any]:
    """Read the YAML config and check required sections."""
    path = find_config(config_path)
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ConfigError(f"{path} is empty or not a YAML mapping")
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ConfigError(f"{path} is missing sections: {missing}")
    if cfg["module"].get("id") != "module_05_ppe":
        raise ConfigError("module.id must be exactly 'module_05_ppe'")
    for key in ("gloves_required", "safety_shoes_required"):
        if cfg["ppe_requirements"].get(key):
            logger.warning("ppe_requirements.%s is true but that class is not supported; ignored", key)
    cfg["_config_file"] = str(path)
    return cfg
