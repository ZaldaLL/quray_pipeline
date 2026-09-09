#!/usr/bin/env python3
"""
Step 11 — RANSAC plane fitting per segment.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from quray_pipeline.config import CLUSTERING_DIR, OUTPUT_DIR, PLANE_MAX_DISTANCE, PLANE_MIN_POINTS
from quray_pipeline.utils.io_utils import load_npz, pcd_stem, save_npz, save_result_txt
from quray_pipeline.utils.plane_utils import fit_plane


def main() -> None:
    data = load_npz(OUTPUT_DIR / "step09_all_planes.npz")
    all_planes = data["all_planes"]
    if all_planes.size == 0:
        print("No planes to fit.")
        return

    t0 = time.perf_counter()
    parameters, orientations = fit_plane(
        all_planes,
        max_distance=PLANE_MAX_DISTANCE,
        min_points=PLANE_MIN_POINTS,
    )
    elapsed = time.perf_counter() - t0
    print(f"[TIME] Discontinuity orientation extraction (plane fitting) elapsed {elapsed:.2f}s")

    save_npz(
        OUTPUT_DIR / "step11_plane_fit.npz",
        parameters=parameters,
        orientations=orientations,
    )
    print(f"Fitted {parameters.shape[0]} planes.")

    # Main output (4): orientation extraction + plane fitting params -> pcd_name-orientation/parameters-num_planes.txt
    num_planes = parameters.shape[0]
    if num_planes > 0:
        stem = pcd_stem(OUTPUT_DIR)

        # orientations columns: [dip_angle, dip_direction, plane_id, set_id]; no header
        save_result_txt(
            CLUSTERING_DIR / f"{stem}-orientation-{num_planes}.txt",
            orientations,
            fmt=["%.2f", "%.2f", "%d", "%d"],
        )

        par_table = np.column_stack([parameters[:, 4], parameters[:, 0:4]])
        save_result_txt(
            CLUSTERING_DIR / f"{stem}-parameters-{num_planes}.txt",
            par_table,
            fmt=["%d", "%.6f", "%.6f", "%.6f", "%.6f"],
        )


if __name__ == "__main__":
    main()
