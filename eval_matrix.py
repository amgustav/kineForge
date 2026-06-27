from __future__ import annotations

import argparse
from pathlib import Path

from kineforge.gates import list_gate_profiles
from kineforge.eval_artifacts import write_eval_artifacts
from kineforge.matrix import (
    build_matrix_summary,
    build_replay_index,
    list_matrix_presets,
    load_matrix_preset,
    parse_scenarios,
    write_matrix_report_html,
    write_matrix_summary_csv,
)
from kineforge.reports import copy_file, prepare_run_dir, timestamp, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one kineForge policy across a named eval matrix.")
    parser.add_argument("--policy", default="runs/latest/policy.zip")
    parser.add_argument("--preset", default="default", help="Matrix preset name from configs/eval_matrices.")
    parser.add_argument("--gate", default=None, help="Gate profile name from configs/gates.")
    parser.add_argument("--list-presets", action="store_true", help="List matrix and gate presets, then exit.")
    parser.add_argument(
        "--scenario",
        action="append",
        help="Named scenario as name=failure_a,failure_b. Use name= for no failures. Repeat for multiple scenarios.",
    )
    parser.add_argument("--task", default="tabletop_reach")
    parser.add_argument("--robot", default="arm_v0")
    parser.add_argument("--reward", default="reach_v0")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_presets:
        print("Matrix presets:")
        for preset in list_matrix_presets():
            print(f"  {preset}")
        print("Gate profiles:")
        for profile in list_gate_profiles():
            print(f"  {profile}")
        return
    matrix_preset = load_matrix_preset(args.preset)
    gate_profile = args.gate or matrix_preset.gate_profile
    scenarios = parse_scenarios(args.scenario, preset_name=args.preset)
    run_timestamp = timestamp()
    matrix_dir = prepare_run_dir("eval-matrix", run_timestamp)

    source_policy_path = Path(args.policy)
    policy_snapshot_path = matrix_dir / "policy.zip"
    copy_file(source_policy_path, policy_snapshot_path)

    source_train_metadata = source_policy_path.parent / "train_metadata.json"
    if source_train_metadata.exists():
        copy_file(source_train_metadata, matrix_dir / "train_metadata.json")

    scenario_results = {}
    for scenario in scenarios:
        scenario_dir = matrix_dir / "scenarios" / scenario.name
        result = write_eval_artifacts(
            output_dir=scenario_dir,
            policy_path=policy_snapshot_path,
            source_policy_path=source_policy_path,
            robot=args.robot,
            task=args.task,
            reward=args.reward,
            failures=scenario.failures,
            episodes=args.episodes,
            seed=args.seed,
            run_timestamp=run_timestamp,
            gate_profile=gate_profile,
        )
        result["scenario"] = scenario
        scenario_results[scenario.name] = result

    summary_path = matrix_dir / "matrix_summary.json"
    replay_index_path = matrix_dir / "replay_index.json"
    report_path = matrix_dir / "report.html"
    csv_path = matrix_dir / "summary.csv"
    summary = build_matrix_summary(
        policy_path=policy_snapshot_path,
        robot=args.robot,
        task=args.task,
        reward=args.reward,
        seed=args.seed,
        episodes=args.episodes,
        run_timestamp=run_timestamp,
        scenario_results=scenario_results,
        matrix_preset=matrix_preset.name,
        gate_profile=gate_profile,
    )
    replay_index = build_replay_index(scenario_results)
    write_json(summary_path, summary)
    write_json(replay_index_path, replay_index)
    write_matrix_report_html(report_path, summary, replay_index)
    write_matrix_summary_csv(csv_path, summary, replay_index)

    print(f"Matrix dir: {matrix_dir}")
    print(f"Preset: {matrix_preset.name}")
    print(f"Gate profile: {gate_profile}")
    print(f"Summary JSON: {summary_path}")
    print(f"Replay index JSON: {replay_index_path}")
    print(f"HTML report: {report_path}")
    print(f"CSV summary: {csv_path}")
    for name, scenario in summary["scenarios"].items():
        print(f"{name}: {scenario['gate_status']} ({scenario['scorecard_json']})")


if __name__ == "__main__":
    main()
