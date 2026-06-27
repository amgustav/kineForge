from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from kineforge.reports import timestamp, write_json

REGISTRY_FIELDS = (
    "run_id",
    "kind",
    "path",
    "timestamp",
    "gate_status",
    "success_rate",
    "mean_final_distance",
    "scenario_count",
    "variant_count",
)

RUN_TIMESTAMP_PATTERN = re.compile(r"(\d{8}-\d{6})$")


def _run_timestamp(run_dir: Path, payload: dict[str, Any]) -> str:
    raw_timestamp = payload.get("timestamp")
    if raw_timestamp:
        return str(raw_timestamp)
    match = RUN_TIMESTAMP_PATTERN.search(run_dir.name)
    return match.group(1) if match else ""


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def _summary_metrics(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    return {
        "success_rate": summary.get("success_rate"),
        "mean_final_distance": summary.get("mean_final_distance"),
    }


def summarize_run_dir(run_dir: Path) -> dict[str, Any] | None:
    if not run_dir.is_dir():
        return None

    matrix_summary = _load_json(run_dir / "matrix_summary.json")
    if matrix_summary is not None:
        return {
            "run_id": run_dir.name,
            "kind": "eval_matrix",
            "path": str(run_dir),
            "timestamp": _run_timestamp(run_dir, matrix_summary),
            "gate_status": "PASS" if matrix_summary.get("gate", {}).get("fail_count") == 0 else "FAIL",
            "success_rate": matrix_summary.get("aggregate", {}).get("success_rate"),
            "mean_final_distance": matrix_summary.get("aggregate", {}).get("mean_final_distance"),
            "scenario_count": matrix_summary.get("scenario_count", ""),
            "variant_count": "",
        }

    sweep_summary = _load_json(run_dir / "sweep_summary.json")
    if sweep_summary is not None:
        return {
            "run_id": sweep_summary.get("run_id", run_dir.name),
            "kind": "sweep",
            "path": str(run_dir),
            "timestamp": _run_timestamp(run_dir, sweep_summary),
            "gate_status": "PASS" if sweep_summary.get("gate", {}).get("fail_count") == 0 else "FAIL",
            "success_rate": "",
            "mean_final_distance": "",
            "scenario_count": "",
            "variant_count": sweep_summary.get("variant_count", ""),
        }

    scorecard = _load_json(run_dir / "scorecard.json")
    if scorecard is not None:
        metrics = _summary_metrics(scorecard.get("summary"))
        return {
            "run_id": run_dir.name,
            "kind": "eval" if run_dir.name.startswith("eval") else "latest",
            "path": str(run_dir),
            "timestamp": _run_timestamp(run_dir, scorecard),
            "gate_status": scorecard.get("gate", {}).get("status", ""),
            "success_rate": metrics.get("success_rate"),
            "mean_final_distance": metrics.get("mean_final_distance"),
            "scenario_count": "",
            "variant_count": "",
        }

    train_metadata = _load_json(run_dir / "train_metadata.json")
    if train_metadata is not None:
        return {
            "run_id": run_dir.name,
            "kind": "train" if run_dir.name.startswith("train") else "latest",
            "path": str(run_dir),
            "timestamp": _run_timestamp(run_dir, train_metadata),
            "gate_status": "",
            "success_rate": "",
            "mean_final_distance": "",
            "scenario_count": "",
            "variant_count": "",
        }

    return None


def build_run_index(runs_dir: Path) -> dict[str, Any]:
    rows = []
    if runs_dir.exists():
        for child in sorted(runs_dir.iterdir(), key=lambda path: path.name):
            row = summarize_run_dir(child)
            if row is not None:
                rows.append(row)
    rows.sort(key=lambda row: str(row.get("timestamp") or row["run_id"]), reverse=True)
    return {
        "generated_at": timestamp(),
        "runs_dir": str(runs_dir),
        "run_count": len(rows),
        "runs": rows,
    }


def write_run_index_json(path: Path, index: dict[str, Any]) -> None:
    write_json(path, index)


def write_run_index_csv(path: Path, index: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        for row in index["runs"]:
            writer.writerow({field: row.get(field, "") for field in REGISTRY_FIELDS})
