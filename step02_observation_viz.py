#!/usr/bin/env python3
"""
Step 02 — Generate Fibonacci observation points and visualize.

Offline visualisation output:
  - output/visualization/step02_observation/
    - normals_with_obs_combined.ply  ← main file (normals blue + observation points magenta, combined)
    - observation_points.json
    - step02_summary.json
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from quray_pipeline.config import NUM_OBS_POINTS, OUTPUT_DIR
from quray_pipeline.utils.geometry import fibonacci_sphere
from quray_pipeline.utils.io_utils import load_npz, save_npz
from quray_pipeline.utils.plotting import save_figure, setup_matplotlib
from quray_pipeline.utils.visualization import VisualizationManager


def main() -> None:
    data = load_npz(OUTPUT_DIR / "step01_sampled.npz")
    normals = data["normals"]

    obs_normals = fibonacci_sphere(NUM_OBS_POINTS)
    save_npz(OUTPUT_DIR / "step02_observation.npz", obs_normals=obs_normals)

    # ---- Offline visualisation: unified output to output/visualization/step02_observation/ ----
    vm = VisualizationManager("step02_observation")

    # Combined PLY (recommended): normals (blue) + observation points (magenta)
    all_pts = np.vstack([normals, obs_normals])
    all_colors = np.vstack([
        np.full((normals.shape[0], 3), [0.0, 0.0, 1.0]),      # blue (normals)
        np.full((obs_normals.shape[0], 3), [1.0, 0.0, 1.0]),   # magenta (observation points)
    ])
    vm.save_point_cloud(all_pts, "normals_with_obs_combined", colors=all_colors)
    vm.save_json("observation_points", {
        "num_observation_points": NUM_OBS_POINTS,
        "method": "fibonacci_sphere",
        "obs_normals": obs_normals.tolist(),
    })

    vm.save_json("step02_summary", {
        "n_normals": int(normals.shape[0]),
        "n_observation_points": NUM_OBS_POINTS,
        "main_ply": "normals_with_obs_combined.ply  ← open this file for the full view",
    })

    # ---- Original matplotlib figure (backward compatibility) ----
    setup_matplotlib()
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(normals[:, 0], normals[:, 1], normals[:, 2], "b.", markersize=1, label="Dataset")
    for j in range(obs_normals.shape[0]):
        ax.plot(
            obs_normals[j, 0],
            obs_normals[j, 1],
            obs_normals[j, 2],
            "^",
            markersize=10,
            markerfacecolor="magenta",
            markeredgecolor="k",
            linewidth=2,
        )
    ax.set_xlabel("X (m)", fontsize=18)
    ax.set_ylabel("Y (m)", fontsize=18)
    ax.set_zlabel("Z (m)", fontsize=18)
    ax.legend(loc="best", fontsize=16)
    save_figure(1, fig)
    # Also copy to the visualisation output directory
    vm.save_matplotlib_figure(fig, "fig01_matplotlib")
    plt.close(fig)
    print(f"Observation points: {NUM_OBS_POINTS}")


if __name__ == "__main__":
    main()
