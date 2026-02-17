"""Read / write fMRI configuration files (YAML and JSON).

Port of ``R/config.R``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import ConfigError


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ConfigError(
            "PyYAML is required to read YAML config files. "
            "Install with: pip install pyyaml"
        ) from exc
    with open(path) as fh:
        result: dict[str, Any] = yaml.safe_load(fh) or {}
        return result


def _read_json(path: Path) -> dict[str, Any]:
    with open(path) as fh:
        result: dict[str, Any] = json.load(fh)
        return result


def read_fmri_config(
    file_name: str | Path,
    base_path: str | Path | None = None,
) -> dict[str, Any]:
    """Read an fMRI configuration file.

    Supports ``.yaml`` / ``.yml`` and ``.json`` formats.

    Parameters
    ----------
    file_name : str or Path
        Path to the configuration file.
    base_path : str, Path, or None
        Override for the ``base_path`` field.
    """
    path = Path(file_name)

    if not path.exists():
        raise ConfigError(f"Config file not found: {path}", parameter="file_name")

    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        data = _read_yaml(path)
    elif suffix == ".json":
        data = _read_json(path)
    else:
        raise ConfigError(
            f"Unsupported config format: {suffix}. Use .yaml or .json",
            parameter="file_name",
        )

    defaults: dict[str, Any] = {
        "cmd_flags": "",
        "jobs": 1,
        "base_path": ".",
        "output_dir": "stat_out",
    }
    config = {**defaults, **data}

    if base_path is not None:
        config["base_path"] = str(base_path)

    return config


def write_fmri_config(
    config: dict[str, Any],
    file_name: str | Path,
) -> None:
    """Write an fMRI configuration to a YAML file.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    file_name : str or Path
        Output path (should end in ``.yaml``).
    """
    try:
        import yaml
    except ImportError as exc:
        raise ConfigError(
            "PyYAML is required to write config files. "
            "Install with: pip install pyyaml"
        ) from exc

    # Strip non-serialisable keys
    to_write = {k: v for k, v in config.items() if k not in ("design", "class")}

    with open(file_name, "w") as fh:
        yaml.dump(to_write, fh, default_flow_style=False)
