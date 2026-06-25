from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def config_path(kind: str, name: str) -> Path:
    return project_root() / "configs" / kind / f"{name}.yaml"


def load_named_config(kind: str, name: str) -> dict[str, Any]:
    return load_yaml(config_path(kind, name))


def load_env_configs(
    robot: str,
    task: str,
    reward: str = "reach_v0",
    failures: str = "basic_failures",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        load_named_config("robots", robot),
        load_named_config("tasks", task),
        load_named_config("rewards", reward),
        load_named_config("failures", failures),
    )


def write_config_snapshot(output_dir: Path, configs: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(configs), handle, sort_keys=True)
