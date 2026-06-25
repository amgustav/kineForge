from __future__ import annotations

import numpy as np
import pytest

from kineforge.config import load_env_configs
from kineforge.envs import TabletopReachEnv
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
