from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from kineforge.config import config_path, load_named_config, project_root

DEFAULT_GATE_PROFILE = "standard"
GATE_PROFILE_FIELDS = ("min_success_rate", "max_mean_final_distance", "max_timeout_rate")


@dataclass(frozen=True)
class GateProfile:
    name: str
    description: str
    thresholds: dict[str, float | str | None]
    path: Path | None = None


def list_gate_profiles() -> list[str]:
    gates_dir = project_root() / "configs" / "gates"
    if not gates_dir.exists():
        return []
    return sorted(path.stem for path in gates_dir.glob("*.yaml") if path.is_file())


def load_gate_profile(name: str = DEFAULT_GATE_PROFILE) -> GateProfile:
    profile_name = str(name).strip() or DEFAULT_GATE_PROFILE
    payload = load_named_config("gates", profile_name)
    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise ValueError(f"gate profile {profile_name!r} must define thresholds")
    missing_fields = [field for field in GATE_PROFILE_FIELDS if field not in thresholds]
    if missing_fields:
        raise ValueError(f"gate profile {profile_name!r} missing thresholds: {', '.join(missing_fields)}")
    profile = GateProfile(
        name=str(payload.get("name", profile_name)).strip() or profile_name,
        description=str(payload.get("description", "")),
        thresholds={field: thresholds[field] for field in GATE_PROFILE_FIELDS},
        path=config_path("gates", profile_name),
    )
    resolve_gate_thresholds(profile, success_threshold=0.05)
    return profile


def resolve_gate_thresholds(profile: GateProfile, success_threshold: float) -> dict[str, float]:
    resolved: dict[str, float] = {}
    for key, raw_value in profile.thresholds.items():
        if raw_value in (None, "task_success_threshold"):
            if key != "max_mean_final_distance":
                raise ValueError(f"{profile.name} threshold {key} cannot use task_success_threshold")
            value = float(success_threshold)
        else:
            value = float(raw_value)
        resolved[key] = value

    if not 0.0 <= resolved["min_success_rate"] <= 1.0:
        raise ValueError(f"gate profile {profile.name!r} min_success_rate must be between 0 and 1")
    if resolved["max_mean_final_distance"] < 0.0:
        raise ValueError(f"gate profile {profile.name!r} max_mean_final_distance must be non-negative")
    if not 0.0 <= resolved["max_timeout_rate"] <= 1.0:
        raise ValueError(f"gate profile {profile.name!r} max_timeout_rate must be between 0 and 1")
    return resolved


def build_gate_result(
    *,
    profile: GateProfile,
    success_threshold: float,
    success_rate: float,
    mean_final_distance: float,
    timeout_rate: float,
) -> dict[str, Any]:
    thresholds = resolve_gate_thresholds(profile, success_threshold)
    criteria = {
        "success_rate": {
            "operator": ">=",
            "threshold": thresholds["min_success_rate"],
            "value": success_rate,
            "passed": bool(success_rate >= thresholds["min_success_rate"]),
        },
        "mean_final_distance": {
            "operator": "<=",
            "threshold": thresholds["max_mean_final_distance"],
            "value": mean_final_distance,
            "passed": bool(mean_final_distance <= thresholds["max_mean_final_distance"]),
        },
        "timeout_rate": {
            "operator": "<=",
            "threshold": thresholds["max_timeout_rate"],
            "value": timeout_rate,
            "passed": bool(timeout_rate <= thresholds["max_timeout_rate"]),
        },
    }
    failed_criteria = [name for name, criterion in criteria.items() if not criterion["passed"]]
    status = "PASS" if not failed_criteria else "FAIL"
    if failed_criteria:
        explanation = f"FAIL: {profile.name} gate failed {', '.join(failed_criteria)}."
    else:
        explanation = f"PASS: all {profile.name} gate criteria passed."
    return {
        "status": status,
        "profile": profile.name,
        "profile_description": profile.description,
        "thresholds": thresholds,
        "criteria": criteria,
        "failed_criteria": failed_criteria,
        "explanation": explanation,
    }
