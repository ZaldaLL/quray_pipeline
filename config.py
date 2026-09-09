"""Pipeline configuration."""
from pathlib import Path

# Project root (parent of quray_pipeline/)
ROOT = Path(__file__).resolve().parent.parent

# Data paths (relative to ROOT)
PCD_PATH = ROOT / "dataset" / "Ouray" / "Ouray.pcd"
FALLBACK_PCD_PATH = ROOT / "dataset" / "Ouray" / "down_Ouray.pcd"

# Output directories
OUTPUT_DIR = ROOT / "pipeline_output"
FIGURE_DIR = ROOT / "figures"
CLUSTERING_DIR = ROOT / "Clustering_Results"

# ===================== Offline visualization output directory =====================
VISUALIZATION_DIR = ROOT / "output" / "visualization"
# Multi-view camera positions (Open3D coordinate system, for offscreen PNG snapshots)
CAMERA_VIEWS = {
    "front":    {"front": (0, 0, -1), "lookat": (0, 0, 0), "up": (0, 1, 0)},
    "back":     {"front": (0, 0, 1),  "lookat": (0, 0, 0), "up": (0, 1, 0)},
    "top":      {"front": (0, -1, 0), "lookat": (0, 0, 0), "up": (0, 0, 1)},
    "bottom":   {"front": (0, 1, 0),  "lookat": (0, 0, 0), "up": (0, 0, 1)},
    "left":     {"front": (1, 0, 0),  "lookat": (0, 0, 0), "up": (0, 1, 0)},
    "right":    {"front": (-1, 0, 0), "lookat": (0, 0, 0), "up": (0, 1, 0)},
    "isometric": {"front": (1, 1, 1), "lookat": (0, 0, 0), "up": (0, 1, 0)},
}
# Offscreen PNG width (height auto-proportioned)
OFFSCREEN_PNG_WIDTH = 1600
# Default views saved for offline visualization
DEFAULT_VIEWS = ["front", "top", "left", "isometric"]
# ================================================================

# Sampling
MAX_POINTS = 60_000
ALPHA_INIT = 0.7
ALPHA_STEP = 0.03
RANDOM_SEED = 0  # rng('default') in MATLAB

# Observation points
NUM_OBS_POINTS = 3  # fibonacci_sphere(3)

# GMM — default torchGMM; full-data k=1 incremental fit, "elbow" K by BIC relative improvement (stable, close to MATLAB)
GMM_BACKEND = "torchgmm"   # "torchgmm" | "sklearn"
MAX_GMM_COMPONENTS = 14     # scan upper bound (joint sets generally <= 10, 14 is sufficient and faster)
GMM_REPLICATES = 6          # initializations per k (best-of-N; too low may occasionally hit bad local optima)
GMM_REGULARIZATION = 1e-5
GMM_MAX_EPOCHS = 300        # epoch ceiling
GMM_CONVERGENCE_TOL = 1e-4
GMM_USE_GPU = True
GMM_GPU_ID = 0             # fixed single GPU to avoid result instability from multi-GPU selection
# Multi-process parallel: one sub-process per observation point, sharing the same GPU
# (each fit sets its own seed; results unchanged).
# 0/auto = auto-detect based on number of observation points (default = #obs); 1 = sequential (old behavior).
GMM_NUM_WORKERS = 0
GMM_RANDOM_SEED = 0        # torch/numpy global seed for reproducibility
# Elbow criterion: scan k=1..MAX, select the k where the last relative improvement >= threshold
# Relative improvement = (BIC[k-1]-BIC[k])/|BIC[k-1]|; increasing → favours smaller K, decreasing → favours larger K
# Setting 5e-3 ignores overfitting jumps below 0.3%-0.4% (e.g. obs2 noise spike at k=13)
GMM_KNEE_REL_IMPROVE = 5e-3

# Density peaks
DENSITY_PERCENT = 2.0
DENSITY_THRESHOLD_RATIO = 0.25
LOG_LIKELIHOOD_FRACTION = 1.1  # floor(n / 1.1)

# Inflection / knee detection (movmean + second-order finite differences)
INFLECTION_D2_PEAK_FACTOR = 1.5

# DBSCAN
DBSCAN_MIN_CLUSTER = 100
DBSCAN_KS = [6, 10, 15, 20, 25, 30]

# Plane fitting
PLANE_MAX_DISTANCE = 0.05
PLANE_MIN_POINTS = 20

# Partial visualization filter
PARTIAL_SET_ID = 4

# Matplotlib
FIGURE_DPI = 150

# Interactive popup windows: pop up rotatable 3D windows for angle selection before saving PNGs
# Default off (leave off on headless servers). Enable by either:
#   1) Setting True here;
#   2) Environment variable QURAY_SHOW_FIGURES=1 (run_all.py --show sets this automatically).
# By default only shows 3D-axis figures; to also show 2D figures, set QURAY_SHOW_ALL=1 (or run_all --show-all).

# ===================== Visualization switches =====================
# Whether to pop up a local Open3D visualization window (for GUI debugging)
# True: opens 3D window if a desktop environment is available; silently skipped on headless servers
SHOW_O3D_WINDOW = False

# Whether to export PLY point cloud files (reloadable offline, for local GUI viewing)
# True: all visualization functions output PLY files to ply_out/
EXPORT_PLY = True

# PLY global parameters
PLY_OUTPUT_DIR = ROOT / "ply_out"
# ==========================================================
