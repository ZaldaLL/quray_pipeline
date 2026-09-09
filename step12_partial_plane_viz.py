#!/usr/bin/env python3
"""
Step 12 — Partial plane visualization for a selected joint set.

Offline visualisation output:
  - output/visualization/step12_partial_planes/
    - partial_planes_set{N}_colored.ply  ← main file (specified joint set, hsv colormap)
    - planes/plane_*.ply                  (individual planes)
    - partial_planes_set{N}_meta.json
  - ply_out/   (PLY copies)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from quray_pipeline.config import OUTPUT_DIR, PARTIAL_SET_ID
from quray_pipeline.utils.io_utils import load_npz
from quray_pipeline.utils.plotting import save_figure, setup_matplotlib
from quray_pipeline.utils.visualization import VisualizationManager


def main() -> None:
    data = load_npz(OUTPUT_DIR / "step09_all_planes.npz")
    all_planes = data["all_planes"]
    if all_planes.size == 0:
        print("No planes to visualize.")
        return

    X = all_planes[:, 0:3]
    labels = all_planes[:, 7].astype(int)
    sets = all_planes[:, 6].astype(int)

    valid = sets == PARTIAL_SET_ID
    Xv = X[valid]
    labelsv = labels[valid]
    unique_labels = np.unique(labelsv)
    n_clusters = len(unique_labels)
    colors = plt.cm.hsv(np.linspace(0, 1, max(n_clusters, 1)))

    # ---- Original matplotlib figure ----
    setup_matplotlib()
    fig = plt.figure(figsize=(14, 9), facecolor="w")
    ax = fig.add_subplot(111, projection="3d")

    color_idx = 0
    for plane_id in unique_labels:
        idx = labelsv == plane_id
        if np.sum(idx) < 50:
            continue
        ax.scatter(
            Xv[idx, 0],
            Xv[idx, 1],
            Xv[idx, 2],
            s=10,
            c=[colors[color_idx % len(colors)]],
            edgecolors="none",
        )
        center = Xv[idx].mean(axis=0)
        ax.text(
            center[0],
            center[1],
            center[2] + 12,
            str(int(plane_id)),
            fontsize=10,
            fontweight="bold",
            color="black",
            ha="center",
            va="center",
        )
        color_idx += 1

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("DBSCAN plane segmentation (random colours + plane ID labels)")
    ax.grid(True)
    save_figure(110, fig, matplotlib_show=False)
    plt.close(fig)

    # ---- Unified offline visualisation (PLY + JSON + NPZ) ----
    vm = VisualizationManager("step12_partial_planes")
    vm.save_planes_colored(
        all_planes,
        name="partial_planes",
        by="plane",
        min_points=50,
        joint_set_id=PARTIAL_SET_ID,
        export_per_plane=True,
        metadata={
            "step": 12,
            "joint_set_id": PARTIAL_SET_ID,
            "description": f"Partial plane visualization for joint set {PARTIAL_SET_ID}",
            "columns": ["x", "y", "z", "nx", "ny", "nz", "set_id", "plane_id"],
        },
    )

    # Save full raw data NPZ
    X_raw = all_planes[:, 0:3]
    sets_raw = all_planes[:, 6].astype(int)
    valid = (sets_raw == PARTIAL_SET_ID) & (all_planes[:, 7].astype(int) != -1)
    vm.save_npz(
        "partial_planes_raw",
        points=X_raw,
        normals=all_planes[:, 3:6],
        set_ids=all_planes[:, 6].astype(int),
        plane_ids=all_planes[:, 7].astype(int),
        joint_set_id=PARTIAL_SET_ID,
    )

    # Save colormap mapping
    from quray_pipeline.utils.visualization import get_colormap_mapping
    labels_raw = all_planes[:, 7].astype(int)
    cmap_mapping = get_colormap_mapping(labels_raw[valid], style="hsv")
    vm.save_json("partial_planes_colormap", {
        "colormap": "hsv",
        "mapping": cmap_mapping,
        "joint_set_id": PARTIAL_SET_ID,
    })


if __name__ == "__main__":
    main()
