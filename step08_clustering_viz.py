#!/usr/bin/env python3
"""
Step 08 — Clustering visualization.

Offline visualisation output:
  - output/visualization/step08_clustering/
    - points_colored.ply                  ← main file 1 (original points coloured by cluster)
    - fig90_normals_with_centers.ply      ← main file 2 (coloured normals + green centres)
    - step08_clustering.json
  - ply_out/   (PLY copies)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from quray_pipeline.config import OUTPUT_DIR
from quray_pipeline.utils.io_utils import load_npz
from quray_pipeline.utils.plotting import save_figure, setup_matplotlib
from quray_pipeline.utils.visualization import VisualizationManager, colormap_for_labels


def main() -> None:
    data = load_npz(OUTPUT_DIR / "step01_sampled.npz")
    labels_data = load_npz(OUTPUT_DIR / "step07_labels.npz")

    original_points = data["original_points"]
    original_normals = data["original_normals"]
    labels = labels_data["labels"]
    center_points = labels_data["center_points"]
    num_clusters = center_points.shape[0]

    colors = plt.cm.tab10(np.linspace(0, 1, max(num_clusters, 1)))

    # ---- Offline visualisation (unified output to output/visualization/step08_clustering/) ----
    vm = VisualizationManager("step08_clustering")

    # 1) Save original points coloured by cluster (PLY) — using 'lines' colormap
    vm.save_point_cloud(original_points, "points_colored", labels=labels, colormap="lines")

    # 2) Save normals coloured by cluster (PLY) — on the unit sphere
    vm.save_point_cloud(original_normals, "normals_colored", labels=labels, colormap="lines")

    # 3) Save cluster centres (PLY) — green highlight
    if center_points.size:
        vm.save_centers_highlight(
            center_points, "centers",
            color=(0.0, 1.0, 0.0),
            marker_size_hint=150.0,
        )

    # 4) Save raw data NPZ (preserves all labels, points, normals, centres)
    vm.save_npz(
        "step08_raw",
        original_points=original_points,
        original_normals=original_normals,
        labels=labels,
        center_points=center_points,
    )

    # 5) Save colormap mapping (for local viewer to reconstruct original colours)
    from quray_pipeline.utils.visualization import get_colormap_mapping
    cmap_mapping = get_colormap_mapping(labels, style="lines")
    vm.save_json("step08_colormap", {"colormap": "lines", "mapping": cmap_mapping})

    # 6) Combined PLY: normals (coloured) + centres (green)
    if center_points.size:
        n_nrm = original_normals.shape[0]
        all_pts = np.vstack([original_normals, center_points])
        # Green colour for centre points
        nrm_colors = colormap_for_labels(labels, style="lines")
        ctr_colors = np.full((center_points.shape[0], 3), [0.0, 1.0, 0.0])
        all_colors = np.vstack([nrm_colors, ctr_colors])
        vm.save_point_cloud(all_pts, "fig90_normals_with_centers", colors=all_colors)

    # 7) Metadata
    cluster_sizes = {int(j): int(np.sum(labels == j)) for j in range(num_clusters)}
    vm.save_json("step08_clustering", {
        "num_clusters": int(num_clusters),
        "total_points": int(original_points.shape[0]),
        "cluster_sizes": cluster_sizes,
        "colormap": "lines",
    })

    # ---- Original matplotlib figures ----
    # Figure 1 (fig90): Joint set partitioning — points coloured by cluster
    setup_matplotlib()
    fig = plt.figure(figsize=(12, 9), facecolor="w")
    ax = fig.add_subplot(111, projection="3d")
    for j in range(num_clusters):
        seg = np.where(labels == j)[0]
        if seg.size == 0:
            continue
        ax.scatter(
            original_points[seg, 0],
            original_points[seg, 1],
            original_points[seg, 2],
            s=8,
            c=[colors[j]],
            alpha=0.75,
        )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Joint set partitioning - Point cloud ({num_clusters} sets)")
    ax.grid(True)
    save_figure(90, fig, matplotlib_show=False)
    plt.close(fig)

    # Figure 2 (fig91): Normals + cluster centres
    setup_matplotlib()
    fig2 = plt.figure(figsize=(12, 9), facecolor="w")
    ax2 = fig2.add_subplot(111, projection="3d")
    for j in range(num_clusters):
        seg = np.where(labels == j)[0]
        if seg.size == 0:
            continue
        ax2.scatter(
            original_normals[seg, 0],
            original_normals[seg, 1],
            original_normals[seg, 2],
            s=6,
            c=[colors[j]],
            alpha=0.6,
        )
    if center_points.shape[0]:
        ax2.scatter(
            center_points[:, 0],
            center_points[:, 1],
            center_points[:, 2],
            s=150,
            c="lime",
            edgecolors="k",
            linewidths=1.5,
            depthshade=False,
            label="Cluster centres",
        )
        ax2.legend(loc="best")
    ax2.set_xlabel("X (dimensionless)")
    ax2.set_ylabel("Y (dimensionless)")
    ax2.set_zlabel("Z (dimensionless)")
    ax2.set_title(f"Joint set partitioning - Normals ({num_clusters} sets)")
    ax2.grid(True)
    save_figure(91, fig2, matplotlib_show=False)
    plt.close(fig2)


if __name__ == "__main__":
    main()
