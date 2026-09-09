#!/usr/bin/env python3
"""
Step 09 — Enhanced DBSCAN per joint set.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from quray_pipeline.config import DBSCAN_MIN_CLUSTER, OUTPUT_DIR
from quray_pipeline.utils.dbscan_utils import enhanced_dbscan
from quray_pipeline.utils.io_utils import load_pickle, save_npz
from quray_pipeline.utils.profiling import print_memory


def main() -> None:
    res = load_pickle(OUTPUT_DIR / "step07_total_res.pkl")
    total_res = list(res["total_res"])

    clusters = [np.asarray(c, dtype=np.float64) for c in total_res]
    t0 = time.perf_counter()
    all_planes = enhanced_dbscan(clusters, min_cluster=DBSCAN_MIN_CLUSTER)
    print(f"[TIME] Plane segmentation (DBSCAN) elapsed {time.perf_counter() - t0:.2f}s")

    save_npz(OUTPUT_DIR / "step09_all_planes.npz", all_planes=all_planes)
    print(f"DBSCAN output: {all_planes.shape[0]} points across planes.")
    print_memory("step09 done")


if __name__ == "__main__":
    main()
