from __future__ import annotations

import csv
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

from kineforge.config import load_env_configs, load_yaml
from kineforge.evals import build_scorecard
from kineforge.gates import load_gate_profile, list_gate_profiles
from kineforge.matrix import (
    build_matrix_summary,
    build_replay_index,
    compare_summaries,
    load_default_scenarios,
    list_matrix_presets,
    load_matrix_preset,
    parse_scenarios,
    rank_scenario_rows,
    write_matrix_report_html,
    write_matrix_summary_csv,
)
from kineforge.sweeps import (
    SweepConfig,
    SweepVariant,
    build_sweep_summary,
    load_sweep_config,
    merge_config,
    list_sweep_presets,
    rank_variant_rows,
)
from kineforge.envs import TabletopReachEnv
from kineforge.replay import save_distance_over_time_png, save_episode_rewards_png
from kineforge.reports import write_config_snapshot
from kineforge.rewards import compute_reach_reward


def _configs():
    return load_env_configs("arm_v0", "tabletop_reach")


def test_env_reset_step_observation_and_info():
    robot_config, task_config, reward_config, failure_config = _configs()
    env = TabletopReachEnv(
        robot_config,
        task_config,
        reward_config,
        failure_config,
        training=True,
        seed=0,
    )

    obs, _ = env.reset(seed=0)
    assert obs.shape == (13,)
    assert env.action_space.shape == (2,)

    _, reward, _, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert np.isfinite(reward)
    assert {"distance", "success", "timeout", "reward_terms"}.issubset(info)
    assert set(info["reward_terms"]) == {
        "distance",
        "progress",
        "success_bonus",
        "action_penalty",
        "timeout_penalty",
    }
    assert len(env.trajectory) == 2



def test_joint_delta_control_can_reduce_default_distance():
    robot_config, task_config, reward_config, failure_config = _configs()
    env = TabletopReachEnv(
        robot_config,
        task_config,
        reward_config,
        failure_config,
        training=False,
        seed=0,
    )

    _, info = env.reset(seed=0)
    initial_distance = info["distance"]
    for _ in range(5):
        _, _, _, _, info = env.step(np.asarray([-0.5, 1.0], dtype=np.float32))

    assert info["distance"] < initial_distance


def test_eval_failures_change_environment_knobs():
    robot_config, task_config, reward_config, failure_config = _configs()
    env = TabletopReachEnv(
        robot_config,
        task_config,
        reward_config,
        failure_config,
        active_failures={"moved_target", "weak_actuator", "noisy_observation"},
        training=False,
        seed=0,
    )

    _, info = env.reset(seed=0)
    expected_target = np.asarray(task_config["target"]["default_position"], dtype=np.float64) + np.asarray(
        [0.08, -0.05, 0.0],
        dtype=np.float64,
    )
    np.testing.assert_allclose(info["target_position"], expected_target)
    assert info["actuator_scale"] == 1.0

    _, _, _, _, step_info = env.step(np.asarray([1.0, 1.0], dtype=np.float32))
    assert "weak_actuator" in step_info["active_failures"]


def test_reward_terms_match_formula():
    reward, terms = compute_reach_reward(
        distance=0.2,
        previous_distance=0.25,
        success=False,
        action=np.array([1.0, -1.0]),
        timeout=True,
        reward_config={
            "weights": {
                "distance": 1.0,
                "progress": 2.0,
                "success_bonus": 10.0,
                "action_penalty": 0.01,
                "timeout_penalty": 1.0,
            }
        },
    )
    assert reward == pytest.approx(-1.12)
    assert terms["progress"] == pytest.approx(0.1)


def test_cli_seed_arguments_are_parsed(monkeypatch):
    train_cli = importlib.import_module("train")
    eval_cli = importlib.import_module("eval")

    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--task", "tabletop_reach", "--robot", "arm_v0", "--timesteps", "1000", "--seed", "1"],
    )
    train_args = train_cli.parse_args()
    assert train_args.seed == 1
    assert train_args.timesteps == 1000

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval.py",
            "--policy",
            "runs/latest/policy.zip",
            "--failures",
            "moved_target,noisy_observation",
            "--seed",
            "1",
        ],
    )
    eval_args = eval_cli.parse_args()
    assert eval_args.seed == 1
    assert eval_cli.parse_failures(eval_args.failures) == {"moved_target", "noisy_observation"}
    assert eval_args.gate == "standard"

    monkeypatch.setattr(sys, "argv", ["eval.py", "--list-gates"])
    assert eval_cli.parse_args().list_gates is True


def test_eval_matrix_scenario_parser_supports_named_failure_sets():
    scenarios = parse_scenarios(["baseline=", "noisy=noisy_observation, moved_target"])

    assert scenarios[0].name == "baseline"
    assert scenarios[0].failures == ()
    assert scenarios[1].name == "noisy"
    assert scenarios[1].failures == ("moved_target", "noisy_observation")

    default_scenarios = load_default_scenarios()
    assert [scenario.name for scenario in default_scenarios] == [
        "baseline",
        "target_shift",
        "low_friction",
        "high_friction",
        "observation_noise",
        "action_noise",
        "combined_hard",
    ]
    assert default_scenarios[1].failures == ("moved_target",)
    assert default_scenarios[4].failures == ("noisy_observation",)
    assert default_scenarios[5].failures == ("action_noise",)
    assert default_scenarios[6].failures == ("action_noise", "moved_target", "noisy_observation", "weak_actuator")
    assert default_scenarios[2].limitations

    with pytest.raises(ValueError, match="at least two scenarios"):
        parse_scenarios(["baseline="])

    with pytest.raises(ValueError, match="duplicate scenario names"):
        parse_scenarios(["baseline=", "baseline=moved_target"])

    preset = load_matrix_preset("default")
    assert preset.gate_profile == "standard"
    assert "default" in list_matrix_presets()

    ranked = rank_scenario_rows(
        [
            {"scenario": "fail", "gate_status": "FAIL", "success_rate": 1.0, "mean_final_distance": 0.01},
            {"scenario": "pass_far", "gate_status": "PASS", "success_rate": 0.8, "mean_final_distance": 0.04},
            {"scenario": "pass_near", "gate_status": "PASS", "success_rate": 0.8, "mean_final_distance": 0.03},
        ]
    )
    assert [row["scenario"] for row in ranked] == ["pass_near", "pass_far", "fail"]
    assert [row["rank"] for row in ranked] == [1, 2, 3]



def test_sweep_config_loading_resolves_variants_and_overrides(tmp_path):
    sweep_config = load_sweep_config("configs/sweeps/default.yaml", seed_override=7, timesteps_override=11)

    assert sweep_config.name == "default"
    assert [variant.name for variant in sweep_config.variants] == [
        "baseline",
        "distance_heavy",
        "progress_heavy",
    ]
    assert all(variant.seed == 7 for variant in sweep_config.variants)
    assert all(variant.timesteps == 11 for variant in sweep_config.variants)
    assert all(variant.episodes == 3 for variant in sweep_config.variants)

    distance_heavy = sweep_config.variants[1]
    merged_reward = merge_config({"weights": {"distance": 2.0, "progress": 10.0}}, distance_heavy.reward_overrides)
    assert merged_reward["weights"]["distance"] == 2.5
    assert merged_reward["weights"]["progress"] == 10.0
    assert sweep_config.gate_profile == "standard"
    assert all(variant.gate_profile == "standard" for variant in sweep_config.variants)
    assert "default" in list_sweep_presets()
    strict_sweep = load_sweep_config("configs/sweeps/default.yaml", gate_override="strict", seed_override=7, timesteps_override=11)
    assert strict_sweep.gate_profile == "strict"
    assert all(variant.gate_profile == "strict" for variant in strict_sweep.variants)

    malformed_sweep = tmp_path / "malformed_sweep.yaml"
    malformed_sweep.write_text("name: bad\nvariants: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="variants must be a list"):
        load_sweep_config(malformed_sweep)

    malformed_variant = tmp_path / "malformed_variant.yaml"
    malformed_variant.write_text("name: bad\nseed: 1\ntimesteps: 1\nvariants:\n  - not-a-mapping\n  - name: ok\n", encoding="utf-8")
    with pytest.raises(ValueError, match="variant entries must be mappings"):
        load_sweep_config(malformed_variant)


def test_sweep_summary_ranking_prefers_pass_success_then_distance(tmp_path):
    rows = rank_variant_rows(
        [
            {
                "variant": "fail_high_success",
                "gate_status": "FAIL",
                "success_rate": 1.0,
                "mean_final_distance": 0.01,
            },
            {
                "variant": "pass_far",
                "gate_status": "PASS",
                "success_rate": 0.8,
                "mean_final_distance": 0.04,
            },
            {
                "variant": "pass_near",
                "gate_status": "PASS",
                "success_rate": 0.8,
                "mean_final_distance": 0.03,
            },
            {
                "variant": "pass_best_success",
                "gate_status": "PASS",
                "success_rate": 1.0,
                "mean_final_distance": 0.05,
            },
        ]
    )
    assert [row["variant"] for row in rows] == [
        "pass_best_success",
        "pass_near",
        "pass_far",
        "fail_high_success",
    ]
    assert [row["rank"] for row in rows] == [1, 2, 3, 4]

    variants = {
        "pass_best_success": SweepVariant(
            name="pass_best_success",
            robot="arm_v0",
            task="tabletop_reach",
            reward="reach_v0",
            failures=(),
            episodes=1,
            seed=1,
            timesteps=10,
            description="",
            gate_profile="standard",
            task_overrides={},
            reward_overrides={},
        ),
        "fail_high_success": SweepVariant(
            name="fail_high_success",
            robot="arm_v0",
            task="tabletop_reach",
            reward="reach_v0",
            failures=(),
            episodes=1,
            seed=1,
            timesteps=10,
            description="",
            task_overrides={},
            gate_profile="standard",
            reward_overrides={},
        ),
    }
    summary = build_sweep_summary(
        sweep_config=SweepConfig("unit", tmp_path / "sweep.yaml", "standard", "", tuple(variants.values())),
        output_dir=tmp_path,
        run_timestamp="20260626-000000",
        variant_results={
            "pass_best_success": {
                "variant": variants["pass_best_success"],
                "scorecard": {
                    "gate": {"status": "PASS", "profile": "standard", "failed_criteria": [], "explanation": "PASS"},
                    "summary": {
                        "success_rate": 1.0,
                        "mean_final_distance": 0.05,
                        "timeout_rate": 0.0,
                        "mean_episode_reward": 1.0,
                        "collision_rate": 0.0,
                    },
                },
                "artifacts": {"policy_zip": "policy.zip", "config_snapshot_yaml": "config_snapshot.yaml"},
                "scorecard_path": tmp_path / "pass" / "scorecard.json",
                "metadata_path": tmp_path / "pass" / "eval_metadata.json",
            },
            "fail_high_success": {
                "variant": variants["fail_high_success"],
                "scorecard": {
                    "gate": {"status": "FAIL", "profile": "standard", "failed_criteria": ["success_rate"], "explanation": "FAIL"},
                    "summary": {
                        "success_rate": 1.0,
                        "mean_final_distance": 0.01,
                        "timeout_rate": 0.0,
                        "mean_episode_reward": 1.0,
                        "collision_rate": 0.0,
                    },
                },
                "artifacts": {"policy_zip": "policy.zip", "config_snapshot_yaml": "config_snapshot.yaml"},
                "scorecard_path": tmp_path / "fail" / "scorecard.json",
                "metadata_path": tmp_path / "fail" / "eval_metadata.json",
            },
        },
    )
    assert [row["variant"] for row in summary["ranked_variants"]] == ["pass_best_success", "fail_high_success"]
    assert summary["gate"] == {"pass_count": 1, "fail_count": 1}

def test_eval_matrix_summary_replay_index_and_comparison(tmp_path):
    def make_result(name: str, success: bool, final_distance: float):
        scenario_dir = tmp_path / name
        scenario_dir.mkdir()
        artifacts = {
            "policy_zip": str(tmp_path / "policy.zip"),
            "scorecard_json": str(scenario_dir / "scorecard.json"),
            "trajectory_png": str(scenario_dir / "trajectory.png"),
            "distance_over_time_png": str(scenario_dir / "distance_over_time.png"),
            "episode_rewards_png": str(scenario_dir / "episode_rewards.png"),
        }
        for artifact_key in ("trajectory_png", "distance_over_time_png", "episode_rewards_png"):
            Path(artifacts[artifact_key]).touch()
        scorecard = build_scorecard(
            Path(artifacts["policy_zip"]),
            "arm_v0",
            "tabletop_reach",
            "reach_v0",
            seed=1,
            failure_modes=() if name.startswith("baseline") else {"moved_target"},
            episode_results=[
                {
                    "episode": 0,
                    "seed": 1,
                    "success": success,
                    "timeout": not success,
                    "final_distance": final_distance,
                    "episode_reward": 2.0 if success else -1.0,
                    "steps": 5,
                    "target_position": [0.45, 0.0, 0.04],
                    "final_position": [0.44, 0.01, 0.04],
                    "active_failures": [] if name.startswith("baseline") else ["moved_target"],
                }
            ],
            success_threshold=0.05,
        )
        scorecard["artifacts"] = artifacts
        return {
            "scorecard": scorecard,
            "scorecard_path": scenario_dir / "scorecard.json",
            "artifacts": artifacts,
        }

    before_results = {
        "baseline": make_result("baseline", True, 0.04),
        "moved_target": make_result("moved_target", False, 0.09),
    }
    after_results = {
        "baseline": make_result("baseline_after", True, 0.03),
        "moved_target": make_result("moved_target_after", True, 0.04),
    }

    before_summary = build_matrix_summary(
        tmp_path / "policy.zip",
        "arm_v0",
        "tabletop_reach",
        "reach_v0",
        seed=1,
        episodes=1,
        run_timestamp="20260625-000000",
        scenario_results=before_results,
    )
    after_summary = build_matrix_summary(
        tmp_path / "policy.zip",
        "arm_v0",
        "tabletop_reach",
        "reach_v0",
        seed=1,
        episodes=1,
        run_timestamp="20260625-000001",
        scenario_results=after_results,
    )
    replay_index = build_replay_index(before_results)
    comparison = compare_summaries(before_summary, after_summary)
    before_summary["scenarios"]["moved_target"]["description"] = "Moved target stress"
    before_summary["scenarios"]["moved_target"]["limitations"] = ["documented limitation"]
    report_path = tmp_path / "report.html"
    csv_path = tmp_path / "summary.csv"
    write_matrix_report_html(report_path, before_summary, replay_index)
    write_matrix_summary_csv(csv_path, before_summary, replay_index)
    html = report_path.read_text(encoding="utf-8")
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8", newline="")))

    assert "eval-matrix-20260625-000000" in html
    assert "unsafe_action_rate" in html
    assert "Moved target stress" in html
    assert "documented limitation" in html
    assert 'href="moved_target/trajectory.png"' in html
    assert rows[0]["run_id"] == "eval-matrix-20260625-000000"
    assert rows[0]["scenario_count"] == "2"
    assert rows[0]["unsafe_action_rate"] == "n/a"
    assert rows[1]["scenario"] == "moved_target"
    assert rows[1]["description"] == "Moved target stress"
    assert rows[1]["limitation"] == "documented limitation"
    assert rows[1]["replay_path"].endswith("moved_target/trajectory.png")

    assert before_summary["scenario_count"] == 2
    assert before_summary["gate"]["pass_count"] == 1
    assert before_summary["scenarios"]["moved_target"]["scorecard_json"].endswith("scorecard.json")
    assert set(replay_index["scenarios"]) == {"baseline", "moved_target"}
    assert comparison["gate_delta"]["pass_count"] == 1
    assert comparison["scenarios"]["moved_target"]["summary_delta"]["success_rate"] == pytest.approx(1.0)


def test_scorecard_structure_includes_gate_thresholds_and_episode_results():
    episode_results = [
        {
            "episode": 0,
            "seed": 1,
            "success": True,
            "timeout": False,
            "final_distance": 0.04,
            "episode_reward": 3.0,
            "steps": 12,
            "target_position": [0.45, 0.0, 0.04],
            "final_position": [0.44, 0.01, 0.04],
            "active_failures": ["moved_target"],
        },
        {
            "episode": 1,
            "seed": 2,
            "success": False,
            "timeout": True,
            "final_distance": 0.08,
            "episode_reward": 1.0,
            "steps": 50,
            "target_position": [0.45, 0.0, 0.04],
            "final_position": [0.42, 0.03, 0.04],
            "active_failures": ["moved_target"],
        },
    ]
    scorecard = build_scorecard(
        Path("runs/eval-test/policy.zip"),
        "arm_v0",
        "tabletop_reach",
        "reach_v0",
        seed=1,
        failure_modes={"moved_target"},
        episode_results=episode_results,
        success_threshold=0.05,
    )

    assert set(scorecard["summary"]) == {
        "success_rate",
        "mean_final_distance",
        "timeout_rate",
        "mean_episode_reward",
        "collision_rate",
    }
    assert scorecard["gate"]["thresholds"] == {
        "min_success_rate": 0.8,
        "max_mean_final_distance": 0.05,
        "max_timeout_rate": 0.3,
    }
    assert scorecard["gate"]["profile"] == "standard"
    assert "standard" in list_gate_profiles()
    strict_scorecard = build_scorecard(
        Path("runs/eval-test/policy.zip"),
        "arm_v0",
        "tabletop_reach",
        "reach_v0",
        seed=1,
        failure_modes={"moved_target"},
        episode_results=episode_results,
        success_threshold=0.05,
        gate_profile=load_gate_profile("strict"),
    )
    assert strict_scorecard["gate"]["profile"] == "strict"
    assert strict_scorecard["gate"]["thresholds"]["min_success_rate"] == 0.95
    assert "success_rate" in strict_scorecard["gate"]["failed_criteria"]
    assert len(scorecard["per_episode"]) == 2
    assert scorecard["failure_modes"] == ["moved_target"]
    assert "collision_rate_explanation" in scorecard


def test_report_artifact_and_config_snapshot_writers(tmp_path):
    snapshot_path = write_config_snapshot(
        tmp_path,
        {"robot": {"name": "arm_v0"}, "ppo": {"n_steps": 256}},
    )
    assert snapshot_path.exists()

    loaded = load_yaml(snapshot_path)
    assert loaded["robot"]["name"] == "arm_v0"
    assert loaded["ppo"]["n_steps"] == 256

    distance_path = tmp_path / "distance_over_time.png"
    rewards_path = tmp_path / "episode_rewards.png"
    save_distance_over_time_png([0.2, 0.1, 0.04], 0.05, distance_path)
    save_episode_rewards_png([1.0, 2.0], rewards_path)

    assert distance_path.exists()
    assert distance_path.stat().st_size > 0
    assert rewards_path.exists()
    assert rewards_path.stat().st_size > 0
