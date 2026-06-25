from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

from kineforge.config import load_env_configs, load_yaml
from kineforge.evals import build_scorecard
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
