#!/usr/bin/env python3
"""
Local 3D viewer — matplotlib mplot3d style (grey grid box).

Usage:
  python visualize_saved_results.py
  python visualize_saved_results.py --step step06
  python visualize_saved_results.py --step step05 --component 3
  python visualize_saved_results.py --all

Dependencies: pip install open3d numpy matplotlib
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path
from typing import Any

import numpy as np

# ── Backend ─────────────────────────────────────────────────────────────
import matplotlib
# Prefer Qt backend (GPU-accelerated, rotation/zoom much smoother than TkAgg)
for _qb in ["Qt5Agg", "Qt6Agg", "QtAgg"]:
    try:
        matplotlib.use(_qb, force=True)
        print(f"[BACKEND] {_qb}")
        break
    except Exception:
        continue

import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIZ_DIR = ROOT / "output" / "visualization"
DISP_FRAC = 1            # Downsample to 1/N of original point count
DISP_CAP =  1000000         # Prevent freezing on huge point clouds


def discover_steps(viz_dir: Path) -> dict[str, dict]:
    steps: dict[str, dict] = {}
    if not viz_dir.exists():
        return steps
    for d in sorted(viz_dir.iterdir()):
        if not d.is_dir():
            continue
        files = sorted(f for f in d.rglob("*") if f.is_file())
        meta: dict[str, Any] = {}
        for mf in files:
            if mf.suffix == ".json":
                try:
                    with open(mf, encoding="utf-8") as fp:
                        meta.update(json.load(fp))
                except Exception:
                    pass
        steps[d.name] = {"dir": d, "files": files, "metadata": meta}
    return steps


def list_steps(steps: dict) -> None:
    if not steps:
        return print("No steps found.")
    descs = {
        "step02_observation":   "plot3 normals + obs points",
        "step04_gmm_fit":       "plot GMM PDF [2D]",
        "step05_density_peaks": "plot3 subdata + decision diagram",
        "step06_merged_centers":"plot3 merged/raw centres",
        "step08_clustering":    "scatter3 cluster colouring",
        "step10_dbscan_planes": "scatter3 DBSCAN planes",
        "step12_partial_planes":"scatter3 partial joint sets",
    }
    for name in sorted(steps.keys()):
        n_ply = sum(1 for f in steps[name]["files"] if f.suffix == ".ply")
        print(f"  {name} ({n_ply} PLY)  {descs.get(name, '')}")
    print()


# ── Loading ────────────────────────────────────────────────────────────
def _load_ply(path: Path, max_pts: int | None = None):
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points)
    clr = np.asarray(pcd.colors) if pcd.has_colors() else None
    n = pts.shape[0]
    target = max_pts if max_pts is not None else min(n // DISP_FRAC, DISP_CAP)
    if n > target:
        idx = np.random.default_rng(42).choice(n, target, replace=False)
        pts = pts[idx]
        if clr is not None:
            clr = clr[idx]
        print(f"  Load {path.name}: {n} → {target} points (1/{DISP_FRAC})")
    else:
        print(f"  Load {path.name}: {n} points")
    return pts, clr


def _plot_colored(ax, pts: np.ndarray, clr: np.ndarray | None,
                  markersize: float = 1, alpha: float = 0.7):
    """Plot by colour groups using plot() — ~10x faster than scatter()."""
    kwargs = dict(markersize=markersize, alpha=alpha, rasterized=True)
    if clr is None:
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                ".", color="blue", **kwargs)
        return
    # Quantise nearby colours into groups
    q = (np.clip(clr, 0, 1) * 15).astype(int)
    codes = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
    for code in np.unique(codes):
        mask = codes == code
        if mask.sum() < 5:
            continue
        ax.plot(pts[mask, 0], pts[mask, 1], pts[mask, 2],
                ".", color=tuple(clr[mask][0]), **kwargs)


def _split_by_color(pts: np.ndarray, clr: np.ndarray | None,
                    target: tuple, tol=0.05):
    """Split: returns (background pts, background colours, target pts) or (pts, clr, None)."""
    if clr is None:
        return pts, None, None
    tgt = np.array(target, dtype=np.float64).reshape(1, 3)
    mask = np.linalg.norm(clr - tgt, axis=1) < tol
    if mask.sum() == 0:
        return pts, clr, None
    return pts[~mask], clr[~mask], pts[mask]


# ── Plotting ───────────────────────────────────────────────────────────
def _new_fig(title: str):
    matplotlib.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(12, 9), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=18)
    # Grey grid box
    for p in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        p.fill = True
        p.set_facecolor("#eaeaea")
        p.set_edgecolor("#cccccc")
    ax.grid(True, color="#b0b0b0", linewidth=0.5)
    # Equal aspect ratio (sphere stays spherical)
    ax.set_box_aspect([1, 1, 1])
    return fig, ax


def _uniform_ticks(ax, spacing: float | None = None):
    """Uniform ticks for all axes. When spacing=None, each axis auto-adapts to ~5 ticks; use 0.25 for dimensionless unit sphere."""

    def _auto_step(lo, hi, target=5):
        span = hi - lo
        if span <= 0:
            return 1.0
        raw = span / (target - 1)
        pow10 = 10 ** np.floor(np.log10(raw))
        mant = raw / pow10
        if mant <= 1.5:
            step = pow10
        elif mant <= 3:
            step = 2 * pow10
        elif mant <= 7:
            step = 5 * pow10
        else:
            step = 10 * pow10
        return round(step) if step >= 1 else step

    for get_lim, set_ticks in [
        (ax.get_xlim3d, ax.set_xticks),
        (ax.get_ylim3d, ax.set_yticks),
        (ax.get_zlim3d, ax.set_zticks),
    ]:
        lo, hi = get_lim()
        sp = spacing if spacing is not None else _auto_step(lo, hi)
        ticks = np.arange(np.floor(lo / sp) * sp, hi + sp * 0.5, sp)
        set_ticks(ticks)


def _on_scroll(event):
    """Mouse wheel zoom (mimics Open3D behaviour)."""
    ax = event.inaxes
    if ax is None or ax.name != "3d":
        return
    scale = 0.9 if event.button == "up" else 1.1
    ax.set_xlim3d([v * scale for v in ax.get_xlim3d()])
    ax.set_ylim3d([v * scale for v in ax.get_ylim3d()])
    ax.set_zlim3d([v * scale for v in ax.get_zlim3d()])
    ax.figure.canvas.draw_idle()


def _show(ax=None, spacing=None):
    if ax is not None:
        _uniform_ticks(ax, spacing=spacing)
    fig = plt.gcf()
    fig.canvas.mpl_connect("scroll_event", _on_scroll)
    plt.show(block=True)


# ── Individual steps ──────────────────────────────────────────────────

def show_step02(viz_dir: Path, steps: dict) -> None:
    d = viz_dir / "step02_observation"
    if not d.exists():
        return print("[ERROR]")
    print(f"\n{'=' * 60}\nStep 02 — Observation points\n{'=' * 60}")

    # Load observation point coordinates from JSON (ensure they are not lost due to downsampling)
    obs_json = d / "observation_points.json"
    obs_pts: np.ndarray | None = None
    if obs_json.exists():
        with open(obs_json, encoding="utf-8") as f:
            obs_data = json.load(f)
        obs_arr = np.array(obs_data.get("obs_normals", []))
        if obs_arr.size > 0:
            obs_pts = obs_arr.reshape(-1, 3)

    # Load normal vector background (extract blue points from merged PLY, or downsample and plot)
    ply = d / "normals_with_obs_combined.ply"
    if ply.exists():
        pts, clr = _load_ply(ply)
        # Remove magenta points (observation points), keep only normals background
        bg, _, _ = _split_by_color(pts, clr, (1.0, 0.0, 1.0))
    else:
        bg = None

    fig, ax = _new_fig("Step 02 — Normals (blue) + Obs points (magenta ▲)")
    if bg is not None and bg.size > 0:
        ax.plot(bg[:, 0], bg[:, 1], bg[:, 2],
                ".", color="#1f77b4", markersize=0.8, alpha=0.4, label="Dataset")
    if obs_pts is not None and obs_pts.shape[0] > 0:
        ax.scatter(obs_pts[:, 0], obs_pts[:, 1], obs_pts[:, 2],
                   c="magenta", s=120, marker="^", edgecolors="k",
                   linewidths=1.5, label=f"Observation points ({obs_pts.shape[0]})")
    ax.set_xlabel("X (m)", fontsize=14)
    ax.set_ylabel("Y (m)", fontsize=14)
    ax.set_zlabel("Z (m)", fontsize=14)
    ax.legend(loc="best", fontsize=12)
    _show(ax, spacing=0.25)


def show_step04(viz_dir: Path, steps: dict) -> None:
    d = viz_dir / "step04_gmm_fit"
    if not d.exists():
        return print("[ERROR]")
    print(f"\n{'=' * 60}\nStep 04 — GMM Fit [2D]\n{'=' * 60}")
    jf = d / "gmm_data.json"
    if jf.exists():
        with open(jf, encoding="utf-8") as f:
            gd = json.load(f)
        print(f"  K={gd.get('best_k')}  BIC={gd.get('bic','?')}")
        for c in gd.get("components", []):
            print(f"  Component {c['component']}: μ={c['mu']:.4f} σ={c['sigma']:.4f}")
        for s in gd.get("separations", []):
            print(f"  {s['peak_pair']}: {s['separation_sigma']:.2f}σ")
    for p in sorted(d.glob("*.png")):
        try:
            webbrowser.open(str(p.resolve()))
        except Exception:
            print(f"  Figure: {p.resolve()}")


def show_step05(viz_dir: Path, steps: dict, component: int | None = None) -> None:
    d = viz_dir / "step05_density_peaks"
    if not d.exists():
        return
    comps = sorted(d.glob("component_*"))
    if not comps:
        return
    print(f"\n{'=' * 60}\nStep 05 — Density Peaks ({len(comps)} components)\n{'=' * 60}")
    if component is not None:
        return _show_comp(comps[component - 1], component)
    for cd in comps:
        mf = cd / "subdata_meta.json"
        if mf.exists():
            with open(mf, encoding="utf-8") as f:
                m = json.load(f)
            print(f"  {cd.name}: {m.get('n_subdata','?')} pts, "
                  f"{m.get('n_cluster_centers',0)} candidate centres, mu={m.get('mean','?'):.4f}")
    print(f"  Number (1-{len(comps)}), a=all, Enter=Skip: ", end="")
    ans = sys.stdin.readline().strip().lower()
    if ans == "a":
        for i, cd in enumerate(comps, 1):
            _show_comp(cd, i)
            if i < len(comps):
                print("  Continue? (Y/n/q): ", end="")
                if sys.stdin.readline().strip().lower() == "q":
                    break
    elif ans.isdigit():
        idx = int(ans)
        if 1 <= idx <= len(comps):
            _show_comp(comps[idx - 1], idx)


def _show_comp(cd: Path, idx: int) -> None:
    dd = cd / "decision_diagram_data.json"
    extra = ""
    if dd.exists():
        with open(dd, encoding="utf-8") as f:
            d = json.load(f)
        extra = f" dc={d.get('dc','?'):.4f}"
    fig, ax = _new_fig(f"Step 05 Component {idx} — subData(black) + Candidate centres(red){extra}")
    if (cd / "subdata.ply").exists():
        pts, _ = _load_ply(cd / "subdata.ply")
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                "k.", markersize=0.8, alpha=0.4)
    if (cd / "cluster_centers.ply").exists():
        ctr, _ = _load_ply(cd / "cluster_centers.ply", 500)
        ax.scatter(ctr[:, 0], ctr[:, 1], ctr[:, 2],
                   c="red", s=150, marker="o", edgecolors="darkred",
                   linewidths=2, label=f"Candidate centres ({ctr.shape[0]})")
    ax.set_xlabel("X (m)", fontsize=14)
    ax.set_ylabel("Y (m)", fontsize=14)
    ax.set_zlabel("Z (m)", fontsize=14)
    ax.legend(loc="best", fontsize=12)
    _show(ax, spacing=0.25)
    png = cd / "decision_diagram.png"
    if png.exists():
        try:
            webbrowser.open(str(png.resolve()))
        except Exception:
            pass


def show_step06(viz_dir: Path, steps: dict) -> None:
    d = viz_dir / "step06_merged_centers"
    if not d.exists():
        return
    print(f"\n{'=' * 60}\nStep 06 — Merged centres\n{'=' * 60}")
    mf = d / "step06_summary.json"
    if mf.exists():
        with open(mf, encoding="utf-8") as f:
            m = json.load(f)
        print(f"  Normals: {m.get('n_normals')} | Raw centres: {m.get('n_raw_centers')} | "
              f"Merged: {m.get('n_merged_centers')}")

    # Load centre points from individual PLY files (avoid centre points being discarded by merged PLY downsampling)
    merged_ctr: np.ndarray | None = None
    raw_ctr: np.ndarray | None = None
    mctr_path = d / "merged_centers.ply"
    rctr_path = d / "all_raw_centers.ply"
    if mctr_path.exists():
        merged_ctr, _ = _load_ply(mctr_path, max_pts=500)
    if rctr_path.exists():
        raw_ctr, _ = _load_ply(rctr_path, max_pts=500)

    for fname, fig, desc, ctr_pts in [
        ("fig40_normals_with_merged_centers.ply", 40, "Merged centres", merged_ctr),
        ("fig80_normals_with_raw_centers.ply", 80, "Raw candidate centres", raw_ctr),
    ]:
        fp = d / fname
        if not fp.exists() and ctr_pts is None:
            continue
        print(f"\n  [Fig{fig}] {desc} (red ●)")

        # Background normals: load from merged PLY (downsampling does not affect background appearance)
        bg = None
        if fp.exists():
            pts, clr = _load_ply(fp)
            bg, _, _ = _split_by_color(pts, clr, (1.0, 0.0, 0.0))

        fig_title = f"Step 06 Fig{fig} — Normals + {desc} (red)"
        fig, ax = _new_fig(fig_title)
        bg_c = "gray" if fig == 40 else "black"
        if bg is not None and bg.size > 0:
            ax.plot(bg[:, 0], bg[:, 1], bg[:, 2],
                    ".", color=bg_c, markersize=0.8, alpha=0.3)
        if ctr_pts is not None and ctr_pts.shape[0] > 0:
            ax.scatter(ctr_pts[:, 0], ctr_pts[:, 1], ctr_pts[:, 2],
                       c="red", s=150, marker="o", edgecolors="darkred",
                       linewidths=2, label=f"{desc} ({ctr_pts.shape[0]})")
        ax.set_xlabel("X (dimensionless)", fontsize=14)
        ax.set_ylabel("Y (dimensionless)", fontsize=14)
        ax.set_zlabel("Z (dimensionless)", fontsize=14)
        ax.legend(loc="best", fontsize=12)
        _show(ax, spacing=0.25)


def show_step08(viz_dir: Path, steps: dict) -> None:
    d = viz_dir / "step08_clustering"
    if not d.exists():
        return
    print(f"\n{'=' * 60}\nStep 08 — Clustering Visualisation\n{'=' * 60}")
    mf = d / "step08_clustering.json"
    if mf.exists():
        with open(mf, encoding="utf-8") as f:
            m = json.load(f)
        print(f"  Clusters: {m.get('num_clusters')} | Total points: {m.get('total_points')}")

    pp = d / "points_colored.ply"
    if pp.exists():
        print("\n  [Point cloud] Coloured by cluster")
        pts, clr = _load_ply(pp)
        fig, ax = _new_fig("Step 08 — Point cloud coloured by cluster")
        _plot_colored(ax, pts, clr)
        ax.set_xlabel("X (m)", fontsize=14)
        ax.set_ylabel("Y (m)", fontsize=14)
        ax.set_zlabel("Z (m)", fontsize=14)
        _show(ax)

    f90 = d / "fig90_normals_with_centers.ply"
    ctr_ply = d / "centers.ply"
    if f90.exists():
        print("\n  [Fig90] Normals (coloured) + Centres (green)")
        pts, clr = _load_ply(f90)
        fig, ax = _new_fig("Step 08 Fig90 — Normals (coloured) + Centres (green)")
        # Background normals: plot per-point by cluster colour
        if pts.size > 0:
            # Separate green centre points (if mixed in)
            bg, bg_clr, _ = _split_by_color(pts, clr, (0.0, 1.0, 0.0), tol=0.05)
            _plot_colored(ax, bg, bg_clr, markersize=1.5, alpha=0.5)
        # Centres: load from individual file (ensure not dropped by downsampling)
        if ctr_ply.exists():
            ctr_pts, _ = _load_ply(ctr_ply, max_pts=100)
            ax.scatter(ctr_pts[:, 0], ctr_pts[:, 1], ctr_pts[:, 2],
                       c="lime", s=200, marker="o", edgecolors="k",
                       linewidths=2, label=f"Cluster centres ({ctr_pts.shape[0]})",
                       depthshade=False)
            print(f"  Green ● = Cluster centres ({ctr_pts.shape[0]})")
        else:
            print("  ⚠ centers.ply missing, cannot display cluster centres")
        ax.set_xlabel("X (dimensionless)", fontsize=14)
        ax.set_ylabel("Y (dimensionless)", fontsize=14)
        ax.set_zlabel("Z (dimensionless)", fontsize=14)
        ax.legend(loc="best", fontsize=12)
        _show(ax, spacing=0.25)


def _browse_planes(planes_dir: Path, title_prefix: str) -> None:
    """Interactive browse of per-plane PLY files (shared by step10/step12)."""
    pfs = sorted(planes_dir.glob("plane_*.ply"))
    if not pfs:
        return
    print(f"\n  Individual planes: {len(pfs)}")
    print("  ID / a=all / Enter=Skip: ", end="")
    ans = sys.stdin.readline().strip().lower()
    if ans == "a":
        for pf in pfs[:20]:
            pts, clr = _load_ply(pf)
            fig, ax = _new_fig(f"{title_prefix} — {pf.stem}")
            _plot_colored(ax, pts, clr)
            ax.set_xlabel("X (m)", fontsize=14)
            ax.set_ylabel("Y (m)", fontsize=14)
            ax.set_zlabel("Z (m)", fontsize=14)
            _show(ax)
    elif ans.isdigit():
        tgt = planes_dir / f"plane_{int(ans):03d}.ply"
        if tgt.exists():
            pts, clr = _load_ply(tgt)
            fig, ax = _new_fig(f"{title_prefix} — plane_{int(ans):03d}")
            _plot_colored(ax, pts, clr)
            ax.set_xlabel("X (m)", fontsize=14)
            ax.set_ylabel("Y (m)", fontsize=14)
            ax.set_zlabel("Z (m)", fontsize=14)
            _show(ax)


def show_step10(viz_dir: Path, steps: dict) -> None:
    d = viz_dir / "step10_dbscan_planes"
    if not d.exists():
        return
    print(f"\n{'=' * 60}\nStep 10 — DBSCAN planes\n{'=' * 60}")
    for mf in sorted(d.glob("*_meta.json")):
        with open(mf, encoding="utf-8") as f:
            m = json.load(f)
        print(f"  {m.get('n_planes')} planes, {m.get('n_points')} points, colormap={m.get('colormap')}")
    cp = d / "all_planes_colored.ply"
    if cp.exists():
        pts, clr = _load_ply(cp)
        fig, ax = _new_fig("Step 10 — DBSCAN plane segmentation (hsv)")
        _plot_colored(ax, pts, clr)
        ax.set_xlabel("X (m)", fontsize=14)
        ax.set_ylabel("Y (m)", fontsize=14)
        ax.set_zlabel("Z (m)", fontsize=14)
        _show(ax)
    planes_dir = d / "planes"
    if planes_dir.exists():
        _browse_planes(planes_dir, "Step 10")


def show_step12(viz_dir: Path, steps: dict) -> None:
    d = viz_dir / "step12_partial_planes"
    if not d.exists():
        return
    print(f"\n{'=' * 60}\nStep 12 — Partial joint sets\n{'=' * 60}")
    sid = "?"
    for mf in sorted(d.glob("*_meta.json")):
        with open(mf, encoding="utf-8") as f:
            m = json.load(f)
        sid = m.get("joint_set_id", "?")
        print(f"  Joint set={sid}, {m.get('n_planes')} planes, {m.get('n_points')} points")

    # Merged coloured overview
    for pp in sorted(d.glob("*_colored.ply")):
        pts, clr = _load_ply(pp)
        fig, ax = _new_fig(f"Step 12 — Joint set {sid} overview")
        _plot_colored(ax, pts, clr)
        ax.set_xlabel("X (m)", fontsize=14)
        ax.set_ylabel("Y (m)", fontsize=14)
        ax.set_zlabel("Z (m)", fontsize=14)
        _show(ax)

    # Per-plane browse
    planes_dir = d / "planes"
    if planes_dir.exists():
        _browse_planes(planes_dir, f"Step 12 — Joint set {sid}")


# ── Menu ──────────────────────────────────────────────────────────────
STEP_ORDER = [
    "step02_observation", "step04_gmm_fit", "step05_density_peaks",
    "step06_merged_centers", "step08_clustering",
    "step10_dbscan_planes", "step12_partial_planes",
]
STEP_FNS = {
    "step02_observation": show_step02, "step04_gmm_fit": show_step04,
    "step05_density_peaks": show_step05, "step06_merged_centers": show_step06,
    "step08_clustering": show_step08, "step10_dbscan_planes": show_step10,
    "step12_partial_planes": show_step12,
}


def show_all_steps(viz_dir: Path, steps: dict) -> None:
    avail = [s for s in STEP_ORDER if s in steps]
    for i, name in enumerate(avail):
        print(f"\n{'─' * 60}\nStep [{i + 1}/{len(avail)}]: {name}")
        STEP_FNS.get(name, lambda a, b: None)(viz_dir, steps)
        if i < len(avail) - 1:
            print("Continue? (Y/n/q): ", end="")
            if sys.stdin.readline().strip().lower() == "q":
                break


def interactive_menu(viz_dir: Path, steps: dict) -> None:
    while True:
        print(f"\n{'=' * 60}")
        print("Quray Pipeline — mplot3d Offline 3D Viewer")
        print(f"{'=' * 60}")
        list_steps(steps)
        avail = sorted(steps.keys())
        for i, n in enumerate(avail, 1):
            print(f"  {i:>2}  — {n}")
        print("  a   — Browse all | q — Quit")
        ans = input("\n> ").strip().lower()
        if ans == "q":
            break
        elif ans == "a":
            show_all_steps(viz_dir, steps)
        else:
            try:
                idx = int(ans) - 1
                if 0 <= idx < len(avail):
                    STEP_FNS.get(avail[idx], lambda a, b: None)(viz_dir, steps)
            except ValueError:
                print("Invalid input.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quray Pipeline mplot3d Offline Browser")
    parser.add_argument("--step", type=str, default=None)
    parser.add_argument("--component", type=int, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dir", type=str, default=None)
    args = parser.parse_args()
    viz_dir = Path(args.dir) if args.dir else DEFAULT_VIZ_DIR
    steps = discover_steps(viz_dir)
    if args.list:
        return list_steps(steps)
    if args.step:
        fn = STEP_FNS.get(args.step)
        if fn:
            if args.step == "step05_density_peaks" and args.component is not None:
                show_step05(viz_dir, steps, component=args.component)
            else:
                fn(viz_dir, steps)
        return
    if args.all:
        return show_all_steps(viz_dir, steps)
    interactive_menu(viz_dir, steps)


if __name__ == "__main__":
    main()



