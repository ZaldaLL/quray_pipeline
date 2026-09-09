# Quray — Discontinuity Orientation Analysis from Point Clouds

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Automated extraction and clustering of rock discontinuity (joint) orientations from
3D point clouds, using **Gaussian Mixture Models** (GMM), **density peaks**
clustering on the unit sphere, and **DBSCAN**-based plane segmentation.

This pipeline is a Python port of the MATLAB `quray.m` workflow, with GPU
acceleration via [torchGMM](https://github.com/jeremymanning/torchgmm) and
interactive 3D visualisation via Open3D and Plotly.

---
![example](./vis.png) 
![example](./vis.png) 
## Features

- **GMM-based orientation modelling** — fits 1D GMMs to angular distances from
  Fibonacci-sphere observation points; selects the optimal number of components
  via BIC elbow detection.
- **Density-peak centre extraction** — per-component density peak analysis on
  the unit sphere to identify candidate dominant orientations.
- **Inflection-threshold centre merging** — second-order finite-difference
  knee detection followed by graph-connected-component merging to remove
  redundant centres (Zobaer et al., 2023).
- **Angular K-means refinement** — iterative spherical K-means with
  bidirectionality-aware sign correction.
- **DBSCAN plane segmentation** — adaptive k-distance knee detection,
  DBSCAN clustering, and RANSAC plane fitting per joint set.
- **Offline visualisation** — exports PLY and matplotlib figures; interactive
  local 3D viewer via `visualize_saved_results.py`.

---

## Repository Structure

```
quray_pipeline/
├── config.py                       # All tunable parameters
├── run_all.py                      # Sequential pipeline runner (steps 1–12)
├── visualize_saved_results.py      # Interactive offline 3D viewer
│
├── step01_load_sample.py           # PCD loading & subsampling
├── step02_observation_viz.py       # Fibonacci observation points
├── step03_gmm_fit.py               # GMM fitting (torchGMM / sklearn)
├── step04_gmm_viz.py               # GMM PDF visualisation
├── step05_extract_centers.py       # Density-peak centre extraction
├── step06_merge_centers.py         # Inflection + graph-based merging
├── step07_label_assignment.py      # Cosine-similarity label assignment
├── step08_clustering_viz.py        # Clustering visualisation
├── step09_dbscan.py                # DBSCAN plane segmentation
├── step10_dbscan_viz.py            # DBSCAN visualisation
├── step11_fit_plane.py             # RANSAC plane fitting
├── step12_partial_plane_viz.py     # Per-joint-set plane visualisation
│
└── utils/
    ├── cluster_utils.py            # Centre merging & angular K-means
    ├── dbscan_utils.py             # DBSCAN with adaptive k-distance
    ├── density_peaks.py            # Angular density-peak clustering
    ├── geometry.py                 # Spherical geometry helpers
    ├── gmm_utils.py                # GMM fitting (torchGMM + sklearn backends)
    ├── io_utils.py                 # NPZ / MAT / JSON / PCD I/O
    ├── plane_utils.py              # RANSAC plane fitting
    ├── plotting.py                 # Matplotlib setup & CJK font support
    ├── profiling.py                # Memory / GPU monitoring
    └── visualization.py            # Offline PLY / HTML / PNG export
```

---

## Installation

### Quick start (pip)

```bash
pip install -r requirements.txt
```

### Full environment (conda)

```bash
conda env create -f environment_viz.yml
conda activate quray-viz
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `numpy`, `scipy` | Numerical computation |
| `torch >= 2.0`, `torchgmm >= 0.1.4` | GPU-accelerated GMM (optional; sklearn fallback) |
| `scikit-learn` | sklearn GMM backend, DBSCAN |
| `open3d >= 0.18` | Point cloud I/O, 3D visualisation |
| `matplotlib >= 3.7` | 2D plotting |
| `psutil` (optional) | Per-step memory monitoring |

---

## Usage

### Full pipeline

```bash
cd /path/to/project-root
python quray_pipeline/run_all.py              # steps 1–12
python quray_pipeline/run_all.py --from 5      # resume from step 5
python quray_pipeline/run_all.py --show        # pop up interactive 3D windows
```

### Individual steps

Each step script can be run independently (VS Code ▶ or command line):

```bash
python quray_pipeline/step01_load_sample.py
python quray_pipeline/step03_gmm_fit.py --backend sklearn --workers 8
python quray_pipeline/step03_gmm_fit.py --backend torchgmm --gpu
```

### Interactive visualisation

```bash
python quray_pipeline/visualize_saved_results.py              # interactive menu
python quray_pipeline/visualize_saved_results.py --step step08  # clustering
python quray_pipeline/visualize_saved_results.py --all          # sequential browse
```

---

## Input Format

- **Point cloud**: `.pcd` file with normals (XYZ + NxNyNz).
- Default path: `dataset/<name>/<name>.pcd` (configure in `config.py` → `PCD_PATH`).
- Edit `config.py` to point to your own point cloud file.

---

## Output Format

| Path | Contents |
|------|----------|
| `pipeline_output/step*.npz` | Intermediate NumPy arrays |
| `Clustering_Results/{K}_clustering_analysis.mat` | Final clustering (MATLAB v7.3) |
| `Clustering_Results/*.txt` | Per-point: `x y z nx ny nz label` |
| `cc.mat` | Merged cluster centres |
| `figures/fig*.png` | Matplotlib diagnostic figures |
| `output/visualization/` | Offline PLY + JSON (per step) |
| `ply_out/` | PLY exports |

---

## Configuration

All tunable parameters are in `config.py`:

| Section | Key Parameters |
|---------|---------------|
| Sampling | `MAX_POINTS`, `ALPHA_INIT`, `RANDOM_SEED` |
| GMM | `GMM_BACKEND`, `MAX_GMM_COMPONENTS`, `GMM_REPLICATES`, `GMM_KNEE_REL_IMPROVE` |
| Density Peaks | `DENSITY_PERCENT`, `DENSITY_THRESHOLD_RATIO` |
| Inflection Detection | `INFLECTION_MODE`, `INFLECTION_D2_PEAK_FACTOR`, `INFLECTION_MOVMEAN_WINDOW_1/2` |
| Angular K-means | `KMEANS_MAX_ITER`, `KMEANS_TOL` |
| DBSCAN | `DBSCAN_MIN_CLUSTER`, `DBSCAN_KS` |
| Plane Fitting | `PLANE_MAX_DISTANCE`, `PLANE_MIN_POINTS` |
| Visualisation | `SHOW_O3D_WINDOW`, `EXPORT_PLY` |

---

## Algorithm Reference

The pipeline implements the method described in the thesis section
*"Candidate Direction Centre Fusion and Dominant Orientation Set
Determination"* (Section 3.3.3).

Key steps:
1. **Angular dissimilarity**: `arccos(|c_i^T c_j|)` — accounts for normal
   vector bidirectionality (n ≡ −n).
2. **Inflection threshold** (Zobaer et al., 2023): second-order finite
   differences on the sorted dissimilarity sequence locate the transition
   between intra-group and inter-group angular distances.
3. **Graph connected components**: centres with pairwise distance ≤ τ_c are
   linked; each connected component forms one dominant orientation group.
4. **Sign correction & averaging**: within each group, normals are flipped
   to a common hemisphere before computing the mean direction.
5. **Angular K-means**: iterative refinement using the merged centres as
   initialisation, with bidirectionality-aware assignment.

---

## Citation

If you use this software in your research, please cite the corresponding
thesis and the original MATLAB `quray.m` workflow. A BibTeX entry will be
added upon publication.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Contributing

Contributions are welcome. Please open a GitHub issue to discuss proposed
changes before submitting a pull request.
