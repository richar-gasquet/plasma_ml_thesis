#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot time snapshots from:
  - data file: "density2_tavg.dat" (rows = time, columns = space)
  - time file: "density1_tavg_times.dat" (1D list of timestamps, length = n_times)

Outputs:
  - One PNG per selected time step saved to a refreshed directory "ccp_pic_data_plots"
"""

from __future__ import annotations
import os
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# ==========================================================
# USER PARAMETERS
# ==========================================================
DATA_FILE = "density2_tavg.dat"        # rows = time, columns = space
TIME_FILE = "density2_tavg_times.dat"  # list of timestamps
OUTDIR = "ccp_pic_data_plots"          # refreshed output directory

STRIDE = 10          # plot every Nth snapshot (1 = every)
MAX_FRAMES = 100   # optional cap on number of frames (None = all)

# Plot appearance
FIG_W, FIG_H = 8.0, 5.0
DPI = 130
LINE_WIDTH = 1.5
Y_LABEL = "Density (a.u.)"
TITLE_PREFIX = "Snapshot:"
TIME_UNITS = ""     # e.g. "µs"
ENABLE_GRID = True
BASE_NAME = "snapshot"

# Axis formatting
LABEL_SIZE = 12
TITLE_SIZE = 13
TICK_SIZE = 11
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = None, None


# ==========================================================
# Main entry point
# ==========================================================
def main():
    data_path = Path(DATA_FILE)
    times_path = Path(TIME_FILE)
    outdir = Path(OUTDIR)

    # 1) Load inputs
    U = load_data_matrix(data_path)  # shape: (nt, nx)
    t = load_times(times_path)       # shape: (nt,)
    nt, nx = U.shape

    # 2) Basic validations
    if t.size != nt:
        raise ValueError(
            f"Time array length ({t.size}) does not match number of rows in data ({nt})."
        )
    if STRIDE < 1:
        raise ValueError("STRIDE must be a positive integer.")

    # 3) Prepare normalized spatial grid
    x = np.linspace(0.0, 1.0, nx, endpoint=True)

    # 4) Refresh output directory
    refresh_dir(outdir)

    # 5) Determine which snapshots to plot
    indices = np.arange(0, nt, STRIDE, dtype=int)
    if MAX_FRAMES is not None:
        indices = indices[:MAX_FRAMES]
    total_plots = len(indices)

    # 6) Generate plots with progress counter
    for i, k in enumerate(indices, start=1):
        y = U[k, :]
        ts = t[k]

        # progress counter (prints on one line)
        print(f"\r[ {i:3d} / {total_plots:3d} ] Plotting snapshot at t = {ts:.6g}", end="", flush=True)

        fig = plt.figure(figsize=(FIG_W, FIG_H))
        ax = fig.add_subplot(111)
        ax.plot(x, y, linewidth=LINE_WIDTH)
        ax.set_xlabel("x (normalized 0 → 1)", fontsize=LABEL_SIZE)
        ax.set_ylabel(Y_LABEL, fontsize=LABEL_SIZE)
        title = f"{TITLE_PREFIX} t = {ts:.6g} {TIME_UNITS}".strip()
        ax.set_title(title, fontsize=TITLE_SIZE)
        if ENABLE_GRID:
            ax.grid(True)
        ax.tick_params(labelsize=TICK_SIZE)
        if Y_MIN is not None or Y_MAX is not None:
            ax.set_ylim(bottom=Y_MIN, top=Y_MAX)
        if X_MIN is not None or X_MAX is not None:
            ax.set_xlim(left=X_MIN, right=X_MAX)

        fname = outdir / f"{BASE_NAME}_{k:05d}.png"
        plt.tight_layout()
        plt.savefig(fname.as_posix(), dpi=DPI)
        plt.close(fig)

    print("\nAll snapshots successfully saved.")


# ==========================================================
# Helper functions
# ==========================================================
def load_data_matrix(path: Path) -> np.ndarray:
    """Load a 2D array from text. Rows = time, columns = space."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    arr = np.loadtxt(path.as_posix())
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def load_times(path: Path) -> np.ndarray:
    """Load a 1D array of timestamps from text (one row or one column)."""
    if not path.exists():
        raise FileNotFoundError(f"Time file not found: {path}")
    t = np.loadtxt(path.as_posix())
    return np.ravel(t)  # ensure 1D


def refresh_dir(path: Path) -> None:
    """Delete and recreate directory."""
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(f"Path exists and is not a directory: {path}")
        shutil.rmtree(path.as_posix())
    os.makedirs(path.as_posix(), exist_ok=True)


# ==========================================================
# Script entry
# ==========================================================
if __name__ == "__main__":
    main()
