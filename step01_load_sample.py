#!/usr/bin/env python3
"""
Step 01 — Load PCD, normalize normals, subsample points.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from quray_pipeline.config import (
    ALPHA_INIT,
    ALPHA_STEP,
    FALLBACK_PCD_PATH,
    MAX_POINTS,
    OUTPUT_DIR,
    PCD_PATH,
    RANDOM_SEED,
)
from quray_pipeline.utils.geometry import normalize_rows
from quray_pipeline.utils.io_utils import load_point_cloud, save_json, save_npz


def main() -> None:
    np.random.seed(RANDOM_SEED)

    original_points, original_normals, used_path = load_point_cloud(PCD_PATH, FALLBACK_PCD_PATH)
    original_normals = normalize_rows(original_normals)

    n = original_points.shape[0]
    alpha = ALPHA_INIT
    size_n = int(round(n * alpha))
    while size_n > MAX_POINTS:
        alpha -= ALPHA_STEP
        size_n = int(round(n * alpha))

    idxs_random = np.random.permutation(n)[:size_n]
    points = original_points[idxs_random]
    normals = original_normals[idxs_random]

    out = OUTPUT_DIR / "step01_sampled.npz"
    save_npz(
        out,
        original_points=original_points,
        original_normals=original_normals,
        points=points,
        normals=normals,
        idxs_random=idxs_random,
    )
    save_json(
        OUTPUT_DIR / "step01_meta.json",
        {
            "pcd_path": str(used_path),
            "n_original": int(n),
            "n_sampled": int(size_n),
            "alpha": float(alpha),
        },
    )
    print(f"Loaded {n} points, sampled {size_n} (alpha={alpha:.3f})")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
