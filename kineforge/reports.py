from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def prepare_run_dir(prefix: str) -> Path:
    run_dir = Path("runs") / f"{prefix}-{timestamp()}"
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


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
