#!/usr/bin/env python3
"""
Step 06 — Merge redundant centers, cluster, visualize, save cc.mat.

Offline visualisation output:
  - output/visualization/step06_merged_centers/
    - fig40_normals_with_merged_centers.ply  ← main file (gray normals + red merged centres)
    - fig80_normals_with_raw_centers.ply      ← (black normals + red raw centres)
    - step06_summary.json
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from quray_pipeline.config import OUTPUT_DIR, ROOT as CFG_ROOT
from quray_pipeline.utils.cluster_utils import get_clusters, get_merged_center_points
from quray_pipeline.utils.io_utils import load_npz, save_mat_v73, save_npz
from quray_pipeline.utils.plotting import save_figure, setup_matplotlib
from quray_pipeline.utils.visualization import VisualizationManager


def main() -> None:
    data = load_npz(OUTPUT_DIR / "step01_sampled.npz")
    centers = load_npz(OUTPUT_DIR / "step05_centers.npz")
    normals = data["normals"]
    all_center_points = centers["all_center_points"]

    center_points = get_merged_center_points(all_center_points)
    center_points = get_clusters(center_points)

    save_npz(OUTPUT_DIR / "step06_center_points.npz", center_points=center_points)
    save_mat_v73(CFG_ROOT / "cc.mat", centerPoints=center_points)

    # ---- Offline visualisation ----
    vm = VisualizationManager("step06_merged_centers")

    # Save normal vectors background (PLY)
    vm.save_point_cloud(normals, "normals")

    # Save merged cluster centres (PLY) — red highlight
    if center_points.size:
        vm.save_centers_highlight(
            center_points, "merged_centers",
            color=(1.0, 0.0, 0.0),
            marker_size_hint=8.0,
        )

    # Save raw candidate centres (PLY) — red highlight
    if all_center_points.size:
        vm.save_centers_highlight(
            all_center_points, "all_raw_centers",
            color=(1.0, 0.0, 0.0),
            marker_size_hint=8.0,
        )

    # Combined PLY: fig40 = normals(gray) + merged_centers(red)
    # Saved as a single file; users can overlay both PLY files in Open3D
    n_normals = normals.shape[0]
    if center_points.size:
        all_pts_40 = np.vstack([normals, center_points])
        all_colors_40 = np.zeros((all_pts_40.shape[0], 3))
        all_colors_40[:n_normals] = [0.6, 0.6, 0.6]  # gray
        all_colors_40[n_normals:] = [1.0, 0.0, 0.0]  # red
        vm.save_point_cloud(all_pts_40, "fig40_normals_with_merged_centers",
                            colors=all_colors_40)

    # Combined PLY: fig80 = normals(black) + raw_centers(red)
    if all_center_points.size:
        all_pts_80 = np.vstack([normals, all_center_points])
        all_colors_80 = np.zeros((all_pts_80.shape[0], 3))
        all_colors_80[:n_normals] = [0.0, 0.0, 0.0]  # black
        all_colors_80[n_normals:] = [1.0, 0.0, 0.0]  # red
        vm.save_point_cloud(all_pts_80, "fig80_normals_with_raw_centers",
                            colors=all_colors_80)

    # Metadata
    vm.save_json("step06_summary", {
        "n_normals": int(n_normals),
        "n_raw_centers": int(all_center_points.shape[0]) if all_center_points.size else 0,
        "n_merged_centers": int(center_points.shape[0]) if center_points.size else 0,
    })

    # ---- Original matplotlib figures ----
    setup_matplotlib()

    fig40 = plt.figure(num=40, figsize=(8, 6))
    ax40 = fig40.add_subplot(111, projection="3d")
    ax40.plot(normals[:, 0], normals[:, 1], normals[:, 2], ".", color=(0.6, 0.6, 0.6), markersize=1)
    if center_points.size:
        ax40.plot(
            center_points[:, 0],
            center_points[:, 1],
            center_points[:, 2],
            "ro",
            markersize=8,
            markerfacecolor="r",
        )
    ax40.grid(True)
    ax40.set_xlabel("X (dimensionless)", fontsize=20)
    ax40.set_ylabel("Y (dimensionless)", fontsize=20)
    ax40.set_zlabel("Z (dimensionless)", fontsize=20)
    save_figure(40, fig40)
    plt.close(fig40)

    fig80 = plt.figure(num=80, figsize=(8, 6))
    ax80 = fig80.add_subplot(111, projection="3d")
    ax80.plot(normals[:, 0], normals[:, 1], normals[:, 2], "k.", markersize=1)
    if all_center_points.size:
        ax80.plot(
            all_center_points[:, 0],
            all_center_points[:, 1],
            all_center_points[:, 2],
            "ro",
            markersize=8,
            markerfacecolor="r",
        )
    ax80.grid(True)
    ax80.set_xlabel("X (dimensionless)", fontsize=20)
    ax80.set_ylabel("Y (dimensionless)", fontsize=20)
    ax80.set_zlabel("Z (dimensionless)", fontsize=20)
    save_figure(80, fig80)
    plt.close(fig80)

    print(f"Merged center points: {center_points.shape[0]}")
    print(f"Saved cc.mat -> {CFG_ROOT / 'cc.mat'}")


if __name__ == "__main__":
    main()
