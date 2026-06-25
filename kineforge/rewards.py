from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def compute_reach_reward(
    distance: float,
    success: bool,
    action: np.ndarray,
    timeout: bool,
    reward_config: Mapping[str, Any],
) -> tuple[float, dict[str, float]]:
    weights = reward_config["weights"]
    terms = {
        "distance": -float(weights["distance"]) * float(distance),
        "success_bonus": float(weights["success_bonus"]) if success else 0.0,
        "action_penalty": -float(weights["action_penalty"]) * float(np.square(action).sum()),
        "timeout_penalty": -float(weights["timeout_penalty"]) if timeout and not success else 0.0,
    }
    return float(sum(terms.values())), terms
