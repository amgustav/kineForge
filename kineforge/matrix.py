from __future__ import annotations

import csv
import json
from html import escape
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from kineforge.config import load_named_config, project_root

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


@dataclass(frozen=True)
class MatrixPreset:
    name: str
    description: str
    gate_profile: str
    scenarios: tuple[EvalScenario, ...]

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
    if not isinstance(raw_scenario, Mapping):
        raise ValueError("matrix scenario entries must be mappings")
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


def list_matrix_presets() -> list[str]:
    preset_dir = project_root() / "configs" / "eval_matrices"
    return sorted(path.stem for path in preset_dir.glob("*.yaml") if path.is_file())


def load_matrix_preset(name: str = DEFAULT_MATRIX_CONFIG) -> MatrixPreset:
    preset_name = str(name).strip() or DEFAULT_MATRIX_CONFIG
    payload = load_named_config("eval_matrices", preset_name)
    raw_scenarios = payload.get("scenarios", ())
    scenarios = validate_scenarios([scenario_from_config(raw_scenario) for raw_scenario in raw_scenarios])
    return MatrixPreset(
        name=str(payload.get("name", preset_name)).strip() or preset_name,
        description=str(payload.get("description", "")),
        gate_profile=str(payload.get("gate_profile", "standard")),
        scenarios=tuple(scenarios),
    )


def load_default_scenarios() -> list[EvalScenario]:
    return list(load_matrix_preset(DEFAULT_MATRIX_CONFIG).scenarios)


def validate_scenarios(scenarios: list[EvalScenario]) -> list[EvalScenario]:
    names = [scenario.name for scenario in scenarios]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        raise ValueError(f"duplicate scenario names: {', '.join(duplicate_names)}")
    if len(scenarios) < 2:
        raise ValueError("eval matrix needs at least two scenarios")
    return scenarios


def parse_scenarios(raw_scenarios: Iterable[str] | None, preset_name: str = DEFAULT_MATRIX_CONFIG) -> list[EvalScenario]:
    if raw_scenarios is None:
        return validate_scenarios(list(load_matrix_preset(preset_name).scenarios))
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
    matrix_preset: str = DEFAULT_MATRIX_CONFIG,
    gate_profile: str = "standard",
) -> dict[str, Any]:
    scenarios: dict[str, Any] = {}
    for name, result in scenario_results.items():
        scorecard = result["scorecard"]
        scenario = result.get("scenario")
        scenario_summary = {
            "failure_modes": scorecard["failure_modes"],
            "gate_status": scorecard["gate"]["status"],
            "gate_failed_criteria": list(scorecard["gate"].get("failed_criteria", ())),
            "gate_explanation": scorecard["gate"].get("explanation", ""),
            "scorecard_json": str(result["scorecard_path"]),
            "summary": scorecard["summary"],
            "physical_metrics": scorecard.get("physical_metrics", {}),
            "failure_mode_metadata": scorecard.get("failure_mode_metadata", {}),
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
    ranked_scenarios = rank_scenario_rows(
        [
            {
                "scenario": name,
                "gate_status": scenario["gate_status"],
                "success_rate": scenario["summary"]["success_rate"],
                "mean_final_distance": scenario["summary"]["mean_final_distance"],
            }
            for name, scenario in scenarios.items()
        ]
    )
    return {
        "timestamp": run_timestamp,
        "policy_path": str(policy_path),
        "task": task,
        "robot": robot,
        "reward": reward,
        "seed": int(seed),
        "episodes_per_scenario": int(episodes),
        "matrix_preset": matrix_preset,
        "gate_profile": gate_profile,
        "scenario_count": len(scenario_values),
        "gate": {
            "pass_count": int(pass_count),
            "fail_count": int(len(scenario_values) - pass_count),
            "pass_rate": float(pass_count / len(scenario_values)) if scenario_values else 0.0,
        },
        "aggregate": aggregate,
        "ranking": ["gate_status", "success_rate desc", "mean_final_distance asc"],
        "ranked_scenarios": ranked_scenarios,
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


def _overall_gate(summary: Mapping[str, Any]) -> str:
    return "PASS" if int(summary["gate"]["fail_count"]) == 0 else "FAIL"


def _rank_key(row: Mapping[str, Any]) -> tuple[int, float, float, str]:
    gate_rank = 0 if row["gate_status"] == "PASS" else 1
    return (gate_rank, -float(row["success_rate"]), float(row["mean_final_distance"]), str(row["scenario"]))


def rank_scenario_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked_rows = [dict(row) for row in sorted(rows, key=_rank_key)]
    for index, row in enumerate(ranked_rows, start=1):
        row["rank"] = index
    return ranked_rows


def _run_id(summary: Mapping[str, Any]) -> str:
    return f"eval-matrix-{summary['timestamp']}"


def _scenario_metric(scenario: Mapping[str, Any], metric: str) -> Any:
    return scenario["summary"].get(metric, "n/a")


def _scenario_limitations(scenario: Mapping[str, Any]) -> str:
    return "; ".join(str(item) for item in scenario.get("limitations", ()))


def _scenario_replay_path(name: str, replay_index: Mapping[str, Any]) -> str:
    replay_artifacts = replay_index.get("scenarios", {}).get(name, {})
    return str(replay_artifacts.get("trajectory_png", ""))


def _ranked_scenario_names(summary: Mapping[str, Any]) -> list[str]:
    ranked = summary.get("ranked_scenarios")
    if not ranked:
        return list(summary["scenarios"].keys())
    return [str(row["scenario"]) for row in ranked]


def _scenario_rank(summary: Mapping[str, Any], name: str) -> Any:
    for row in summary.get("ranked_scenarios", ()):
        if row["scenario"] == name:
            return row["rank"]
    return ""


def _relative_report_link(path_value: str, output_dir: Path) -> str:
    if not path_value:
        return ""
    path = Path(path_value)
    path_abs = path if path.is_absolute() else Path.cwd() / path
    try:
        return path_abs.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        return str(path_value)


def write_matrix_summary_csv(path: Path, summary: Mapping[str, Any], replay_index: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "policy_path",
        "scenario_count",
        "overall_gate",
        "rank",
        "scenario",
        "success_rate",
        "mean_final_distance",
        "collision_rate",
        "unsafe_action_rate",
        "gate",
        "failed_criteria",
        "description",
        "limitation",
        "scorecard_path",
        "replay_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name in _ranked_scenario_names(summary):
            scenario = summary["scenarios"][name]
            writer.writerow(
                {
                    "run_id": _run_id(summary),
                    "policy_path": summary["policy_path"],
                    "scenario_count": summary["scenario_count"],
                    "overall_gate": _overall_gate(summary),
                    "rank": _scenario_rank(summary, name),
                    "scenario": name,
                    "success_rate": _scenario_metric(scenario, "success_rate"),
                    "mean_final_distance": _scenario_metric(scenario, "mean_final_distance"),
                    "collision_rate": _scenario_metric(scenario, "collision_rate"),
                    "unsafe_action_rate": _scenario_metric(scenario, "unsafe_action_rate"),
                    "gate": scenario["gate_status"],
                    "failed_criteria": ",".join(scenario.get("gate_failed_criteria", ())),
                    "description": scenario.get("description", ""),
                    "limitation": _scenario_limitations(scenario),
                    "scorecard_path": scenario["scorecard_json"],
                    "replay_path": _scenario_replay_path(name, replay_index),
                }
            )


def write_matrix_report_html(path: Path, summary: Mapping[str, Any], replay_index: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in _ranked_scenario_names(summary):
        scenario = summary["scenarios"][name]
        scorecard_path = str(scenario["scorecard_json"])
        replay_path = _scenario_replay_path(name, replay_index)
        scorecard_link = _relative_report_link(scorecard_path, path.parent)
        replay_link = _relative_report_link(replay_path, path.parent)
        replay_cell = f'<a href="{escape(replay_link)}">{escape(replay_path)}</a>' if replay_path else ""
        rows.append(
            "<tr>"
            f"<td>{escape(str(_scenario_rank(summary, name)))}</td>"
            f"<th scope=\"row\">{escape(name)}</th>"
            f"<td>{escape(str(_scenario_metric(scenario, 'success_rate')))}</td>"
            f"<td>{escape(str(_scenario_metric(scenario, 'mean_final_distance')))}</td>"
            f"<td>{escape(str(_scenario_metric(scenario, 'collision_rate')))}</td>"
            f"<td>{escape(str(_scenario_metric(scenario, 'unsafe_action_rate')))}</td>"
            f"<td>{escape(str(scenario['gate_status']))}</td>"
            f"<td>{escape(', '.join(scenario.get('gate_failed_criteria', ())))}</td>"
            f"<td>{escape(str(scenario.get('gate_explanation', '')))}</td>"
            f"<td>{escape(str(scenario.get('description', '')))}</td>"
            f"<td>{escape(_scenario_limitations(scenario))}</td>"
            f"<td><a href=\"{escape(scorecard_link)}\">{escape(scorecard_path)}</a></td>"
            f"<td>{replay_cell}</td>"
            "</tr>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>kineForge eval matrix report - {escape(_run_id(summary))}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 2rem; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 0.45rem 0.55rem; text-align: left; vertical-align: top; }}
    th {{ background: #f0f4f8; }}
    .summary {{ display: grid; grid-template-columns: max-content 1fr; gap: 0.35rem 1rem; max-width: 56rem; }}
    .gate-pass {{ color: #0b6b3a; font-weight: 700; }}
    .gate-fail {{ color: #9b1c1c; font-weight: 700; }}
    code {{ background: #f0f4f8; padding: 0.1rem 0.25rem; }}
  </style>
</head>
<body>
  <h1>kineForge eval matrix report</h1>
  <dl class="summary">
    <dt>run_id</dt><dd><code>{escape(_run_id(summary))}</code></dd>
    <dt>policy path</dt><dd><code>{escape(str(summary["policy_path"]))}</code></dd>
    <dt>scenario count</dt><dd>{escape(str(summary["scenario_count"]))}</dd>
    <dt>gate profile</dt><dd><code>{escape(str(summary.get("gate_profile", "standard")))}</code></dd>
    <dt>overall gate</dt><dd class="gate-{escape(_overall_gate(summary).lower())}">{escape(_overall_gate(summary))}</dd>
  </dl>
  <h2>Scenarios</h2>
  <table>
    <thead>
      <tr>
        <th>rank</th>
        <th>scenario</th>
        <th>success_rate</th>
        <th>mean_final_distance</th>
        <th>collision_rate</th>
        <th>unsafe_action_rate</th>
        <th>gate</th>
        <th>failed criteria</th>
        <th>gate explanation</th>
        <th>description</th>
        <th>limitation</th>
        <th>scorecard path</th>
        <th>replay path</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


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
