from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Any, Mapping

import numpy as np


def sample_uniform_range(rng: np.random.Generator, bounds: Sequence[float]) -> float:
    if len(bounds) != 2:
        raise ValueError(f"Expected [low, high] bounds, got {bounds!r}")
    return float(rng.uniform(float(bounds[0]), float(bounds[1])))


def sample_target_position(
    rng: np.random.Generator,
    task_config: Mapping[str, Any],
    training: bool,
) -> np.ndarray:
    if not training:
        return np.asarray(task_config["target"]["default_position"], dtype=np.float64)

    ranges = task_config["randomization"]["target_position"]
    return np.asarray(
        [
            sample_uniform_range(rng, ranges["x"]),
            sample_uniform_range(rng, ranges["y"]),
            sample_uniform_range(rng, ranges["z"]),
        ],
        dtype=np.float64,
    )


def sample_initial_qpos(
    rng: np.random.Generator,
    task_config: Mapping[str, Any],
    training: bool,
) -> np.ndarray:
    if not training:
        return np.asarray([0.0, 0.0], dtype=np.float64)

    ranges = task_config["randomization"]["initial_qpos"]
    return np.asarray(
        [
            sample_uniform_range(rng, ranges["shoulder"]),
            sample_uniform_range(rng, ranges["elbow"]),
        ],
        dtype=np.float64,
    )


def sample_actuator_scale(
    rng: np.random.Generator,
    task_config: Mapping[str, Any],
    training: bool,
) -> float:
    if not training:
        return 1.0
    return sample_uniform_range(rng, task_config["randomization"]["actuator_scale"])


def apply_moved_target(
    target: np.ndarray,
    failure_config: Mapping[str, Any],
    active_failures: Collection[str],
) -> np.ndarray:
    if "moved_target" not in active_failures:
        return np.asarray(target, dtype=np.float64).copy()
    offset = np.asarray(failure_config["moved_target"]["offset"], dtype=np.float64)
    return np.asarray(target, dtype=np.float64) + offset


def observation_noise_std(
    task_config: Mapping[str, Any],
    failure_config: Mapping[str, Any],
    active_failures: Collection[str],
    training: bool,
) -> float:
    std = float(task_config["randomization"].get("observation_noise_std", 0.0)) if training else 0.0
    if "noisy_observation" in active_failures:
        std += float(failure_config["noisy_observation"]["std"])
    return std


def weak_actuator_scale(
    failure_config: Mapping[str, Any],
    active_failures: Collection[str],
) -> float:
    if "weak_actuator" in active_failures:
        return float(failure_config["weak_actuator"]["scale"])
    return 1.0
