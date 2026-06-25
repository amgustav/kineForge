from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def save_trajectory_png(
    trajectory: np.ndarray,
    target_position: np.ndarray,
    final_position: np.ndarray,
    success: bool,
    output_path: Path,
) -> None:
    trajectory = np.asarray(trajectory, dtype=np.float64)
    target_position = np.asarray(target_position, dtype=np.float64)
    final_position = np.asarray(final_position, dtype=np.float64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6, 6))
    axis.plot(trajectory[:, 0], trajectory[:, 1], label="end-effector path")
    axis.scatter(target_position[0], target_position[1], marker="*", s=160, label="target")
    axis.scatter(final_position[0], final_position[1], marker="o", s=80, label="final")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x position (m)")
    axis.set_ylabel("y position (m)")
    axis.grid(True)
    axis.legend()
    axis.set_title(f"kineForge replay: {'SUCCESS' if success else 'FAILURE'}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
