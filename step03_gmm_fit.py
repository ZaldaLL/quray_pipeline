#!/usr/bin/env python3
"""Step 03 — GMM fitting via torchGMM (GPU-accelerated) or sklearn (multi-core CPU)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from quray_pipeline.config import (
    GMM_BACKEND,
    GMM_CONVERGENCE_TOL,
    GMM_GPU_ID,
    GMM_KNEE_REL_IMPROVE,
    GMM_MAX_EPOCHS,
    GMM_NUM_WORKERS,
    GMM_RANDOM_SEED,
    GMM_REGULARIZATION,
    GMM_REPLICATES,
    GMM_USE_GPU,
    MAX_GMM_COMPONENTS,
    OUTPUT_DIR,
)
from quray_pipeline.utils.gmm_utils import fit_all_observation_points, select_best_observation
from quray_pipeline.utils.io_utils import load_npz, save_json, save_pickle
from quray_pipeline.utils.profiling import GpuSmiSampler, print_memory, profile_section


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 03: GMM fitting")
    parser.add_argument("--backend", choices=("torchgmm", "sklearn"), default=GMM_BACKEND)
    parser.add_argument("--gpu", action="store_true", help="Force CUDA")
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU")
    parser.add_argument(
        "--workers", type=int, default=GMM_NUM_WORKERS,
        help="Number of parallel processes (shared GPU); 0=auto (one per obs), 1=serial",
    )
    args = parser.parse_args()

    use_gpu = GMM_USE_GPU
    if args.gpu:
        use_gpu = True
    if args.no_gpu:
        use_gpu = False

    # Global seed — ensures run_all and standalone runs produce identical results
    import numpy as _np
    import torch as _torch
    _np.random.seed(GMM_RANDOM_SEED)
    _torch.manual_seed(GMM_RANDOM_SEED)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(GMM_RANDOM_SEED)

    data = load_npz(OUTPUT_DIR / "step01_sampled.npz")
    obs = load_npz(OUTPUT_DIR / "step02_observation.npz")

    print(f"GMM fitting  backend={args.backend}, workers={args.workers} ...")
    print_memory("step03 start")
    t0 = time.perf_counter()

    with profile_section("step03 GMM fit"), GpuSmiSampler(GMM_GPU_ID) as smi:
        all_best_k, all_models = fit_all_observation_points(
            data["normals"],
            obs["obs_normals"],
            backend=args.backend,
            max_components=MAX_GMM_COMPONENTS,
            replicates=GMM_REPLICATES,
            regularization=GMM_REGULARIZATION,
            max_epochs=GMM_MAX_EPOCHS,
            convergence_tol=GMM_CONVERGENCE_TOL,
            use_gpu=use_gpu,
            knee_rel_improve=GMM_KNEE_REL_IMPROVE,
            num_workers=args.workers,
        )

    print(f"\nStep03 elapsed {time.perf_counter() - t0:.1f}s")
    if smi.peak_used:
        pct = 100.0 * smi.peak_used / smi.total if smi.total else 0.0
        print(f"[MEM] GPU{GMM_GPU_ID} peak VRAM (nvidia-smi): "
              f"{smi.peak_used}/{smi.total} MB ({pct:.0f}%)")
    print_memory("step03 done")

    for obs_idx, k in enumerate(all_best_k):
        if not np.isnan(k):
            print(f"Obs point {obs_idx + 1}: optimal components = {int(k)}")
        else:
            print(f"Obs point {obs_idx + 1}: fit failed")

    sign_idx, orig_idx, best_model = select_best_observation(all_best_k, all_models)
    print(f"\nSelected obs point {orig_idx + 1}, joint sets K = {best_model.num_components}")

    save_pickle(OUTPUT_DIR / "step03_gmm_models.pkl", all_models)
    save_json(
        OUTPUT_DIR / "step03_gmm_meta.json",
        {
            "all_best_k": [None if np.isnan(k) else int(k) for k in all_best_k],
            "sign_idx": sign_idx,
            "orig_idx": orig_idx,
            "best_k": int(best_model.num_components),
            "best_bic": float(best_model.bic),
            "backend": args.backend,
            "use_gpu": use_gpu,
        },
    )
    save_pickle(OUTPUT_DIR / "step03_best_model.pkl", best_model)
    print("Saved GMM results.")


if __name__ == "__main__":
    main()
