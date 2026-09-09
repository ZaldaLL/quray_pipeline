#!/usr/bin/env python3
"""
Step 04 — Visualize best GMM fit and peak separation.

Offline visualisation output:
  - output/visualization/step04_gmm_fit/
    - gmm_fit.png          (GMM PDF fit curve, 2D matplotlib)
    - gmm_data.json        (peak parameters, separation, BIC, etc.)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from quray_pipeline.config import OUTPUT_DIR
from quray_pipeline.utils.geometry import compute_observation_distance
from quray_pipeline.utils.gmm_utils import GMMResult
from quray_pipeline.utils.io_utils import load_npz, load_pickle
from quray_pipeline.utils.plotting import save_figure, setup_matplotlib
from quray_pipeline.utils.visualization import VisualizationManager


def main() -> None:
    data = load_npz(OUTPUT_DIR / "step01_sampled.npz")
    obs = load_npz(OUTPUT_DIR / "step02_observation.npz")
    best_model: GMMResult = load_pickle(OUTPUT_DIR / "step03_best_model.pkl")
    with open(OUTPUT_DIR / "step03_gmm_meta.json", encoding="utf-8") as f:
        meta = json.load(f)

    normals = data["normals"]
    obs_normals = obs["obs_normals"]
    orig_idx = meta["orig_idx"]
    best_k = meta["best_k"]

    print(f"\nBest model: {best_k} Component(s)")

    obs_normal = obs_normals[orig_idx]
    obs_distances = compute_observation_distance(normals, obs_normal)
    x_range = np.linspace(obs_distances.min(), obs_distances.max(), 500)
    y_pdf = best_model.pdf(x_range)

    # ---- Offline visualisation ----
    vm = VisualizationManager("step04_gmm_fit")

    setup_matplotlib()
    fig = plt.figure(num=20, figsize=(10, 6))
    plt.plot(x_range, y_pdf, "g-", linewidth=4, label="GMM Fit")

    cmap = plt.cm.jet(np.linspace(0, 1, best_k))
    components_info = []
    for k in range(best_k):
        mu = best_model.mu[k]
        sigma = best_model.sigma[k]
        weight = best_model.weights[k]
        component_pdf = stats.norm.pdf(x_range, mu, sigma) * weight
        plt.plot(x_range, component_pdf, "-", linewidth=2, color=cmap[k], label=f"Component {k + 1}")
        peak_height = best_model.pdf(np.array([mu]))[0]
        plt.plot(mu, peak_height, "o", markersize=8, markerfacecolor=cmap[k], markeredgecolor="k", linewidth=2)
        components_info.append({
            "component": k + 1,
            "mu": float(mu),
            "sigma": float(sigma),
            "weight": float(weight),
            "peak_height": float(peak_height),
        })

    print("\n=== Peak Separation Analysis ===")
    mus = np.sort(best_model.mu)
    sigmas = best_model.sigma[np.argsort(best_model.mu)]
    separations = []
    for k in range(best_k - 1):
        separation = (mus[k + 1] - mus[k]) / max(sigmas[k], sigmas[k + 1])
        separations.append({
            "peak_pair": f"Peak{k + 1}-Peak{k + 2}",
            "separation_sigma": float(separation),
        })
        print(f"Peak{k + 1}-Peak{k + 2}: separation = {separation:.2f}sigma")

    plt.xlabel("Euclidean distance", fontsize=20)
    plt.ylabel("Probability density", fontsize=20)
    plt.grid(True)
    save_figure(20, fig)
    # Copy to visualisation directory
    vm.save_matplotlib_figure(fig, "gmm_fit")
    plt.close(fig)

    # Export GMM fit data as JSON (for local viewer)
    vm.save_json("gmm_data", {
        "best_k": best_k,
        "orig_idx": orig_idx,
        "bic": float(best_model.bic),
        "observation_point": obs_normal.tolist(),
        "components": components_info,
        "separations": separations,
        "x_range": x_range.tolist(),
        "y_pdf": y_pdf.tolist(),
    })


if __name__ == "__main__":
    main()
