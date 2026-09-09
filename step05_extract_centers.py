#!/usr/bin/env python3
"""
Step 05 — Extract cluster centers per GMM component.

Offline visualisation output:
  - output/visualization/step05_density_peaks/
    - component_01/
      - subdata.ply
      - decision_diagram.png
      - decision_diagram_data.json
      - subdata_meta.json
    - component_02/ ...
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from quray_pipeline.config import (
    DENSITY_PERCENT,
    DENSITY_THRESHOLD_RATIO,
    LOG_LIKELIHOOD_FRACTION,
    OUTPUT_DIR,
)
from quray_pipeline.utils.density_peaks import get_densities_and_distances_for_higher_density_angular
from quray_pipeline.utils.geometry import auto_dc_angular, compute_observation_distance
from quray_pipeline.utils.gmm_utils import GMMResult
from quray_pipeline.utils.io_utils import load_npz, load_pickle, save_npz
from quray_pipeline.utils.plotting import save_figure, setup_matplotlib
from quray_pipeline.utils.profiling import print_memory
from quray_pipeline.utils.visualization import VisualizationManager


def main() -> None:
    data = load_npz(OUTPUT_DIR / "step01_sampled.npz")
    obs = load_npz(OUTPUT_DIR / "step02_observation.npz")
    best_model: GMMResult = load_pickle(OUTPUT_DIR / "step03_best_model.pkl")
    with open(OUTPUT_DIR / "step03_gmm_meta.json", encoding="utf-8") as f:
        meta = json.load(f)

    normals = data["normals"]
    obs_normals = obs["obs_normals"]
    orig_idx = meta["orig_idx"]
    optimal_k = best_model.num_components

    obs_normal = obs_normals[orig_idx]
    distances = compute_observation_distance(normals, obs_normal)

    all_center_points: list[np.ndarray] = []
    plot_index = 1

    # Offline visualisation manager
    vm = VisualizationManager("step05_density_peaks")

    for i in range(optimal_k):
        mean_value = best_model.mu[i]
        variance_value = best_model.sigma[i] ** 2

        loglikelihoods = stats.norm.logpdf(distances, loc=mean_value, scale=np.sqrt(variance_value))
        sorted_index = np.argsort(loglikelihoods)
        start = int(np.floor(len(sorted_index) / LOG_LIKELIHOOD_FRACTION))
        higher_local_density_indices = sorted_index[start:]

        sub_data = normals[higher_local_density_indices]

        # ---- Original matplotlib figure ----
        setup_matplotlib()
        fig = plt.figure(num=plot_index + 1, figsize=(12, 5))

        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.plot(sub_data[:, 0], sub_data[:, 1], sub_data[:, 2], "k.", markersize=1)
        ax1.set_xlabel("X (m)", fontsize=16)
        ax1.set_ylabel("Y (m)", fontsize=16)
        ax1.set_zlabel("Z (m)", fontsize=16)

        dc = auto_dc_angular(sub_data, DENSITY_PERCENT)
        print(f"Computing Rho with gaussian kernel of radius: {dc:12.6f}")

        densities, density_distances, density_indices = get_densities_and_distances_for_higher_density_angular(
            sub_data, dc
        )

        max_density = densities[0]
        max_density_distance = density_distances[0]
        density_threshold = max_density * DENSITY_THRESHOLD_RATIO
        density_distance_threshold = max_density_distance * DENSITY_THRESHOLD_RATIO
        conditions = (densities > density_threshold) & (density_distances > density_distance_threshold)
        condition_indices = np.where(conditions)[0]

        cluster_centers = normals[higher_local_density_indices[density_indices[condition_indices]]]
        if cluster_centers.size:
            all_center_points.append(cluster_centers)
        # None means no candidate centres in this component
        comp_cluster_centers = cluster_centers if cluster_centers.size else None

        # ---- Offline visualisation: per-component subdirectory (PLY + NPZ + JSON) ----
        comp_meta = {
            "component_index": i + 1,
            "mean": float(mean_value),
            "variance": float(variance_value),
            "n_subdata": int(sub_data.shape[0]),
            "percent": DENSITY_PERCENT,
        }
        vm.save_component_visualization(
            i + 1, sub_data, "subdata",
            cluster_centers=comp_cluster_centers,
            metadata=comp_meta,
        )

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(densities, density_distances, "bo")
        ax2.plot(
            [density_threshold, density_threshold],
            [0, max_density_distance],
            "r-",
            linewidth=2.5,
        )
        ax2.plot(
            [0, max_density],
            [density_distance_threshold, density_distance_threshold],
            "r-",
            linewidth=2.5,
        )
        ax2.grid(True)
        ax2.set_xlabel("Density", fontsize=20)
        ax2.set_ylabel("Delta distance", fontsize=20)

        save_figure(plot_index + 1, fig)

        # Copy decision diagram to visualisation directory
        comp_dir = vm.step_dir / f"component_{i + 1:02d}"
        fig.savefig(comp_dir / "decision_diagram.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Save decision diagram data
        dd_path = comp_dir / "decision_diagram_data.json"
        dd_data = {
            "dc": float(dc),
            "density_threshold": float(density_threshold),
            "density_distance_threshold": float(density_distance_threshold),
            "n_cluster_centers": int(cluster_centers.shape[0]) if cluster_centers.size else 0,
            "densities": densities.tolist(),
            "density_distances": density_distances.tolist(),
        }
        if cluster_centers.size:
            dd_data["cluster_centers"] = cluster_centers.tolist()
        with open(dd_path, "w", encoding="utf-8") as f_json:
            json.dump(dd_data, f_json, indent=2, ensure_ascii=False)

        plot_index += 1
        print_memory(f"step05 component {i + 1}/{optimal_k} (N={sub_data.shape[0]})")

    if all_center_points:
        all_center_points_arr = np.vstack(all_center_points)
    else:
        all_center_points_arr = np.zeros((0, 3))

    save_npz(
        OUTPUT_DIR / "step05_centers.npz",
        all_center_points=all_center_points_arr,
        distances=distances,
        higher_indices_base=np.arange(normals.shape[0]),
    )
    print(f"Extracted {all_center_points_arr.shape[0]} raw center candidates.")


if __name__ == "__main__":
    main()
