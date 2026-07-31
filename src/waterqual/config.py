"""Configuration loading and relative-path resolution.

Every script loads the pipeline configuration through :func:`load_config`, which
reads ``config/analysis_config.yaml`` and returns a :class:`Config` wrapper.

Design goals (reproducibility requirements):
  * No machine-specific absolute paths anywhere in the tree. The repo root is
    discovered from this file's location, and every path in the YAML is resolved
    relative to it.
  * Output directories are created automatically on demand.
  * A helper (:meth:`Config.require`) fails with a clear, actionable message when
    an expected input file is missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Repo root = two levels up from this file:  <root>/src/waterqual/config.py
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "analysis_config.yaml"


class ConfigError(RuntimeError):
    """Raised when configuration or a required input is missing/malformed."""


class Config:
    """Thin wrapper around the parsed YAML with path resolution helpers."""

    def __init__(self, data: dict, config_path: Path):
        self._data = data
        self.config_path = config_path
        self.root = REPO_ROOT

    # -- dict-style access ---------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    @property
    def data(self) -> dict:
        return self._data

    # -- path helpers --------------------------------------------------------
    def path(self, key: str) -> Path:
        """Resolve a named entry under ``paths:`` to an absolute path.

        ``key`` may be a top-level key in the ``paths`` block (e.g.
        ``"matrix_scaled"``) or a raw relative path string.
        """
        paths = self._data.get("paths", {})
        rel = paths.get(key, key)
        p = Path(rel)
        return p if p.is_absolute() else (self.root / p)

    def resolve(self, rel: str) -> Path:
        """Resolve any relative path string against the repo root."""
        p = Path(rel)
        return p if p.is_absolute() else (self.root / p)

    def ensure_dir(self, key_or_rel: str) -> Path:
        """Resolve a directory path and create it (parents included)."""
        d = self.path(key_or_rel)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def require(self, key: str, hint: str = "") -> Path:
        """Return an absolute input path, raising a helpful error if absent."""
        p = self.path(key)
        if not p.exists():
            msg = (
                f"Required input not found:\n"
                f"  config key : paths.{key}\n"
                f"  expected at: {p}\n"
            )
            if hint:
                msg += f"  hint       : {hint}\n"
            msg += "  See data/README.md for how to obtain or regenerate this file."
            raise ConfigError(msg)
        return p

    # -- convenience accessors ----------------------------------------------
    @property
    def feature_order(self) -> list[str]:
        return list(self._data["features"]["order"])

    @property
    def random_state(self) -> int:
        return int(self._data["project"]["random_state"])


def load_config(config_path: str | Path | None = None) -> Config:
    """Load the pipeline configuration.

    Parameters
    ----------
    config_path:
        Path to the YAML config. Defaults to ``config/analysis_config.yaml``
        at the repo root.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise ConfigError(
            f"Configuration file not found at {path}. "
            f"Run scripts from the repository, or pass --config explicitly."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(data, path)
