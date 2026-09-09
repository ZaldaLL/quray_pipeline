#!/usr/bin/env python3
"""
Run steps 01-12 sequentially (same pattern as Discontinuousity-new/scripts/run_all.py).

Usage:
    python quray_pipeline/run_all.py
    python quray_pipeline/run_all.py --from 3 --to 7
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None


from quray_pipeline.utils.profiling import format_bytes as _fmt_bytes


def _run_step_monitored(cmd: list[str], cwd: str) -> float | None:
    """Run a single step; poll the sub-process (including its children) for peak RSS via psutil.

    Returns the peak RSS in bytes for the step, or None if psutil is unavailable. VRAM peaks are
    printed by the individual steps themselves (e.g. step03 GMM), because CUDA stats are
    process-local and the parent process cannot read them.
    """
    if psutil is None:
        subprocess.run(cmd, cwd=cwd, check=True)
        return None

    proc = subprocess.Popen(cmd, cwd=cwd)
    ps_proc = psutil.Process(proc.pid)
    peak_rss = 0
    stop = threading.Event()

    def _sample() -> None:
        nonlocal peak_rss
        while not stop.is_set():
            try:
                total = ps_proc.memory_info().rss
                for child in ps_proc.children(recursive=True):
                    try:
                        total += child.memory_info().rss
                    except Exception:
                        pass
                peak_rss = max(peak_rss, total)
            except Exception:
                break
            stop.wait(0.2)

    sampler = threading.Thread(target=_sample, daemon=True)
    sampler.start()
    ret = proc.wait()
    stop.set()
    sampler.join(timeout=1.0)
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)
    return peak_rss

# Stage labels for timing summary (Joint Set Clustering / Plane Segmentation / Orientation Extraction)
STAGES = [
    ("Joint Set Clustering", range(1, 9)),               # steps 1-8
    ("Plane Segmentation (DBSCAN)", range(9, 11)),        # steps 9-10
    ("Discontinuity Orientation Extraction & Plane Fitting", range(11, 13)),  # steps 11-12
]

STEPS = [
    "step01_load_sample.py",
    "step02_observation_viz.py",
    "step03_gmm_fit.py",
    "step04_gmm_viz.py",
    "step05_extract_centers.py",
    "step06_merge_centers.py",
    "step07_label_assignment.py",
    "step08_clustering_viz.py",
    "step09_dbscan.py",
    "step10_dbscan_viz.py",
    "step11_fit_plane.py",
    "step12_partial_plane_viz.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run quray pipeline steps in order.")
    parser.add_argument("--from", dest="from_step", type=int, default=1, help="Start step (1-12)")
    parser.add_argument("--to", dest="to_step", type=int, default=12, help="End step (1-12)")
    parser.add_argument(
        "--show", action="store_true",
        help="Pop up rotatable 3D figure windows (close window to continue); PNGs still saved as usual",
    )
    parser.add_argument(
        "--show-all", dest="show_all", action="store_true",
        help="Also pop up 2D figures (by default only 3D-axis figures are shown)",
    )
    args, extra = parser.parse_known_args()

    if args.show or args.show_all:
        os.environ["QURAY_SHOW_FIGURES"] = "1"
        if args.show_all:
            os.environ["QURAY_SHOW_ALL"] = "1"

    timings: list[tuple[int, str, float, float | None]] = []
    total_t0 = time.perf_counter()

    if psutil is None:
        print("[MEM] psutil not installed, skipping per-step RSS monitoring; consider `pip install psutil`.")

    for i, name in enumerate(STEPS, start=1):
        if i < args.from_step or i > args.to_step:
            continue
        script = PIPELINE / name
        print(f"\n{'=' * 60}\n>>> {name}\n{'=' * 60}")
        cmd = [sys.executable, str(script), *extra]
        t0 = time.perf_counter()
        peak_rss = _run_step_monitored(cmd, cwd=str(ROOT))
        elapsed = time.perf_counter() - t0
        timings.append((i, name, elapsed, peak_rss))
        suffix = f", peak RSS {_fmt_bytes(peak_rss)}" if peak_rss is not None else ""
        print(f"<<< {name} finished in {elapsed:.1f}s{suffix}")

    print("\n--- Timing / memory summary ---")
    max_peak = 0.0
    for _i, name, elapsed, peak_rss in timings:
        mem = f"{_fmt_bytes(peak_rss):>10}" if peak_rss is not None else f"{'N/A':>10}"
        if peak_rss is not None:
            max_peak = max(max_peak, peak_rss)
        print(f"  {elapsed:7.1f}s  peak RSS {mem}  {name}")
    print(f"  {'':7}  TOTAL {time.perf_counter() - total_t0:.1f}s")
    if max_peak:
        print(f"  pipeline peak RSS ≈ {_fmt_bytes(max_peak)} (GPU VRAM peaks shown in per-step [MEM] lines)")

    # Stage timing summary (only for steps that actually ran)
    print("\n--- Stage Timing Summary ---")
    for label, rng in STAGES:
        ran = [(idx, e) for (idx, _n, e, _r) in timings if idx in rng]
        if not ran:
            continue
        secs = sum(e for _idx, e in ran)
        idxs = [idx for idx, _e in ran]
        print(f"  {secs:7.1f}s  {label}  (steps {min(idxs)}-{max(idxs)})")
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
