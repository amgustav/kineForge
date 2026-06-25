from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from kineforge.config import load_named_config

from kineforge.reports import write_json

SCENARIO_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
DEFAULT_MATRIX_CONFIG = "default"
SUMMARY_METRICS = (
    "success_rate",
    "mean_final_distance",
    "timeout_rate",
    "mean_episode_reward",
    "collision_rate",
)
REPLAY_ARTIFACT_KEYS = (
    "trajectory_png",
    "distance_over_time_png",
    "episode_rewards_png",
)


@dataclass(frozen=True)
class EvalScenario:
    name: str
    failures: tuple[str, ...]
    description: str = ""
    limitations: tuple[str, ...] = ()

def parse_failures(raw_failures: str) -> set[str]:
    return {failure.strip() for failure in raw_failures.split(",") if failure.strip()}


def parse_scenario(raw_scenario: str) -> EvalScenario:
    if "=" not in raw_scenario:
        raise ValueError("scenario must use name=failure_a,failure_b format")
    raw_name, raw_failures = raw_scenario.split("=", 1)
    name = raw_name.strip()
    if not name:
        raise ValueError("scenario name cannot be empty")
    if not SCENARIO_NAME_PATTERN.fullmatch(name):
        raise ValueError("scenario names may only contain letters, numbers, underscores, and hyphens")
    failures = tuple(sorted(parse_failures(raw_failures)))
    return EvalScenario(name=name, failures=failures)


def scenario_from_config(raw_scenario: Mapping[str, Any]) -> EvalScenario:
    name = str(raw_scenario["name"]).strip()
    if not name:
        raise ValueError("scenario name cannot be empty")
    if not SCENARIO_NAME_PATTERN.fullmatch(name):
        raise ValueError("scenario names may only contain letters, numbers, underscores, and hyphens")
    failures = tuple(sorted(str(failure).strip() for failure in raw_scenario.get("failures", ()) if str(failure).strip()))
    limitations = tuple(str(item) for item in raw_scenario.get("limitations", ()))
    return EvalScenario(
        name=name,
        failures=failures,
        description=str(raw_scenario.get("description", "")),
        limitations=limitations,
    )


def load_default_scenarios() -> list[EvalScenario]:
    payload = load_named_config("eval_matrices", DEFAULT_MATRIX_CONFIG)
    return [scenario_from_config(raw_scenario) for raw_scenario in payload["scenarios"]]


def validate_scenarios(scenarios: list[EvalScenario]) -> list[EvalScenario]:
    names = [scenario.name for scenario in scenarios]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        raise ValueError(f"duplicate scenario names: {', '.join(duplicate_names)}")
    if len(scenarios) < 2:
        raise ValueError("eval matrix needs at least two scenarios")
    return scenarios


def parse_scenarios(raw_scenarios: Iterable[str] | None) -> list[EvalScenario]:
    if raw_scenarios is None:
        return validate_scenarios(load_default_scenarios())
    return validate_scenarios([parse_scenario(raw) for raw in raw_scenarios])


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def build_matrix_summary(
    policy_path: Path,
    robot: str,
    task: str,
    reward: str,
    seed: int,
    episodes: int,
    run_timestamp: str,
    scenario_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    scenarios: dict[str, Any] = {}
    for name, result in scenario_results.items():
        scorecard = result["scorecard"]
        scenario = result.get("scenario")
        scenario_summary = {
            "failure_modes": scorecard["failure_modes"],
            "gate_status": scorecard["gate"]["status"],
            "scorecard_json": str(result["scorecard_path"]),
            "summary": scorecard["summary"],
        }
        if scenario is not None:
            if scenario.description:
                scenario_summary["description"] = scenario.description
            if scenario.limitations:
                scenario_summary["limitations"] = list(scenario.limitations)
        scenarios[name] = scenario_summary

    scenario_values = list(scenarios.values())
    pass_count = sum(1 for scenario in scenario_values if scenario["gate_status"] == "PASS")
    aggregate = {
        metric: _mean([float(scenario["summary"][metric]) for scenario in scenario_values])
        for metric in SUMMARY_METRICS
    }
    return {
        "timestamp": run_timestamp,
        "policy_path": str(policy_path),
        "task": task,
        "robot": robot,
        "reward": reward,
        "seed": int(seed),
        "episodes_per_scenario": int(episodes),
        "scenario_count": len(scenario_values),
        "gate": {
            "pass_count": int(pass_count),
            "fail_count": int(len(scenario_values) - pass_count),
            "pass_rate": float(pass_count / len(scenario_values)) if scenario_values else 0.0,
        },
        "aggregate": aggregate,
        "scenarios": scenarios,
    }


def build_replay_index(scenario_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    scenarios: dict[str, Any] = {}
    for name, result in scenario_results.items():
        replay_artifacts = {
            key: value
            for key, value in result["artifacts"].items()
            if key in REPLAY_ARTIFACT_KEYS and Path(value).exists()
        }
        if replay_artifacts:
            scenarios[name] = replay_artifacts
    return {"scenarios": scenarios}


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"summary is not a JSON object: {path}")
    if "scenarios" not in payload or "aggregate" not in payload:
        raise ValueError(f"summary is missing matrix fields: {path}")
    return payload


def _metric_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for metric in SUMMARY_METRICS:
        if metric in before and metric in after:
            deltas[metric] = float(after[metric]) - float(before[metric])
    return deltas


def compare_summaries(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    before_scenarios = before["scenarios"]
    after_scenarios = after["scenarios"]
    before_names = set(before_scenarios)
    after_names = set(after_scenarios)
    common_names = sorted(before_names & after_names)

    scenario_deltas = {}
    for name in common_names:
        before_scenario = before_scenarios[name]
        after_scenario = after_scenarios[name]
        scenario_deltas[name] = {
            "before_gate_status": before_scenario["gate_status"],
            "after_gate_status": after_scenario["gate_status"],
            "summary_delta": _metric_delta(before_scenario["summary"], after_scenario["summary"]),
        }

    return {
        "before_policy_path": before.get("policy_path"),
        "after_policy_path": after.get("policy_path"),
        "before_timestamp": before.get("timestamp"),
        "after_timestamp": after.get("timestamp"),
        "added_scenarios": sorted(after_names - before_names),
        "removed_scenarios": sorted(before_names - after_names),
        "aggregate_delta": _metric_delta(before["aggregate"], after["aggregate"]),
        "gate_delta": {
            "pass_count": int(after["gate"]["pass_count"]) - int(before["gate"]["pass_count"]),
            "fail_count": int(after["gate"]["fail_count"]) - int(before["gate"]["fail_count"]),
            "pass_rate": float(after["gate"]["pass_rate"]) - float(before["gate"]["pass_rate"]),
        },
        "scenarios": scenario_deltas,
    }


def compare_summary_files(before_path: Path, after_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    comparison = compare_summaries(load_summary(before_path), load_summary(after_path))
    if output_path is not None:
        write_json(output_path, comparison)
    return comparison
