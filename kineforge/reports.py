from __future__ import annotations

import json
import platform
import shutil
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

import yaml


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def prepare_run_dir(prefix: str, run_timestamp: str | None = None) -> Path:
    if run_timestamp is None:
        run_timestamp = timestamp()
    run_dir = Path("runs") / f"{prefix}-{run_timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def reset_latest_dir() -> Path:
    latest_dir = Path("runs") / "latest"
    if latest_dir.exists():
        if latest_dir.is_symlink() or latest_dir.is_file():
            latest_dir.unlink()
        else:
            shutil.rmtree(latest_dir)
    latest_dir.mkdir(parents=True)
    return latest_dir


def ensure_latest_dir() -> Path:
    latest_dir = Path("runs") / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    return latest_dir


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_config_snapshot(output_dir: Path, configs: Mapping[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / "config_snapshot.yaml"
    with snapshot_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(configs), handle, sort_keys=True)
    return snapshot_path


def package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    packages = {
        "kineforge": "kineforge",
        "mujoco": "mujoco",
        "gymnasium": "gymnasium",
        "stable_baselines3": "stable-baselines3",
        "numpy": "numpy",
        "pyyaml": "PyYAML",
        "matplotlib": "matplotlib",
    }
    for output_key, package_name in packages.items():
        try:
            versions[output_key] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[output_key] = "unavailable"
    return versions


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
