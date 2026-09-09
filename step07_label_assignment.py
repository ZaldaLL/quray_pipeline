#!/usr/bin/env python3
"""
Step 07 — K-medoids label assignment (quray.m lines 293-297).

Scalable implementation: never builds an N×N distance matrix.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from quray_pipeline.config import CLUSTERING_DIR, OUTPUT_DIR
from quray_pipeline.utils.cluster_utils import kmedoids_angular
from quray_pipeline.utils.io_utils import (
    ensure_dir,
    load_npz,
    pcd_stem,
    save_mat_v73,
    save_npz,
    save_pickle,
    save_result_txt,
)


def main() -> None:
    data = load_npz(OUTPUT_DIR / "step01_sampled.npz")
    centers = load_npz(OUTPUT_DIR / "step06_center_points.npz")

    original_points = data["original_points"]
    original_normals = data["original_normals"]
    center_points = centers["center_points"]

    if center_points.shape[0] == 0:
        raise RuntimeError("No center points found. Check previous steps.")

    # K-medoids — scalable, no N×N matrix
    refined_centers, labels = kmedoids_angular(original_normals, center_points)
    center_points = refined_centers  # medoids are actual normals

    num_clusters = center_points.shape[0]
    for i in range(num_clusters):
        print(f"Cluster {i + 1}: {np.sum(labels == i)} normals")

    list_of_class = [np.where(labels == i)[0] for i in range(num_clusters)]  # 0-based Python indices
    list_of_class_matlab = [idx + 1 for idx in list_of_class]  # 1-based for .mat
    total_res: list[np.ndarray] = []

    for i in range(num_clusters):
        cla = list_of_class[i]
        if cla.size == 0:
            continue
        points_ = original_points[cla]
        normals_ = original_normals[cla]
        label_col = np.full((cla.size, 1), i + 1, dtype=np.float64)  # MATLAB 1-based label column
        total_res.append(np.hstack([points_, normals_, label_col]))

    ensure_dir(CLUSTERING_DIR)
    mat_path = CLUSTERING_DIR / f"{num_clusters}_clustering_analysis.mat"
    save_mat_v73(
        mat_path,
        total_res=np.array(total_res, dtype=object),
        centerPoints=center_points,
        labels=labels + 1,  # save 1-based like MATLAB
        list_of_class=np.array(list_of_class_matlab, dtype=object),
    )

    save_npz(
        OUTPUT_DIR / "step07_labels.npz",
        labels=labels,
        center_points=center_points,
    )
    # Variable-length arrays stored via pickle (npz object arrays require allow_pickle)
    save_pickle(
        OUTPUT_DIR / "step07_total_res.pkl",
        {"total_res": total_res, "list_of_class": list_of_class},
    )

    print("Label assignment complete.")
    print(f"Clustering results saved to: {mat_path}")

    # Main output (4): clustering analysis results -> pcd_name-clustering-num_clusters.txt
    # One point per line: x y z nx ny nz label (label is 1-based joint set number)
    if total_res:
        cluster_table = np.vstack(total_res)
        stem = pcd_stem(OUTPUT_DIR)
        txt_path = CLUSTERING_DIR / f"{stem}-clustering-{num_clusters}.txt"
        save_result_txt(
            txt_path,
            cluster_table,
            fmt=["%.6f"] * 6 + ["%d"],
        )


if __name__ == "__main__":
    main()
