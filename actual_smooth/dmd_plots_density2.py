#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 2 — Post-processing & plots for CCP density2 DMD analysis.

What it does (modeled after diffusion_bounded_linear/dmd_plots.py):
1) Loads the NPZ file produced by dmd_analysis_density2.py.
2) Plots time series of:
   - Absolute Frobenius error: ||Yhat - Y||_F
   - Relative Frobenius error: ||Yhat - Y||_F / ||Y||_F
   - Normalized-to-initial error: ||Yhat - Y||_F / ||Y(0)||_F
   All three plots are saved into a refreshed directory "dmd_norm_plots".
3) Plots selected snapshots (truth vs discrete vs continuous) at a user-specified
   frequency; saved in a refreshed directory "dmd_snapshots".
"""

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# User config
# ----------------------------
NPZ_PATH = "dmd_density2_ANALYSIS.npz"

# Output directories (mirroring structure of diffusion_bounded_linear/dmd_plots.py)
BASE_DIR = "dmd_density2_plots"
NORM_PLOTS_DIR = os.path.join(BASE_DIR, "dmd_norm_plots")   # refreshed each run
SNAPSHOT_DIR   = os.path.join(BASE_DIR, "dmd_snapshots")    # refreshed each run

# Snapshot plotting frequency:
#   0 -> plot none
#   k -> plot every k-th snapshot (k >= 1)
SNAPSHOT_FREQ = 50

# Optional: line styles for snapshot overlays
STYLE_TRUE = dict(lw=2, label="Truth")
STYLE_DISC = dict(lw=2, ls="--", label="DMD (discrete)")
STYLE_CONT = dict(lw=2, ls=":",  label="DMD (continuous)")


def main():
    # 1) Load NPZ from Script 1
    if not os.path.exists(NPZ_PATH):
        raise FileNotFoundError(f"Could not find NPZ file: {NPZ_PATH}")

    data = np.load(NPZ_PATH, allow_pickle=True)

    # Time axis (decimated window aligned with comparisons 0..t_max)
    t_dec = data["t_decimated"]

    # --- Error series saved by Script 1 ---
    # Absolute errors
    err_disc_abs_t = data["err_disc_abs_t"]  # ||Yhat_disc - Y||_F
    err_cont_abs_t = data["err_cont_abs_t"]  # ||Yhat_cont - Y||_F
    # Relative errors
    err_disc_rel_t = data["err_disc_rel_t"]  # / ||Y||
    err_cont_rel_t = data["err_cont_rel_t"]
    # Normalized-to-initial errors
    err_disc_norm0_t = data["err_disc_norm0_t"]  # / ||Y(0)||
    err_cont_norm0_t = data["err_cont_norm0_t"]

    # Fields for snapshot comparisons (aligned to t_dec)
    Y_truth   = data["Y_truth"]
    Yhat_disc = data["Yhat_disc"]
    Yhat_cont = data["Yhat_cont"]

    # 2) Error plots → refreshed "dmd_norm_plots"
    refresh_dir(NORM_PLOTS_DIR)

    plot_two_series(
        t=t_dec,
        y1=err_disc_abs_t,
        y2=err_cont_abs_t,
        label1="||E_disc||_F",
        label2="||E_cont||_F",
        ylabel="Absolute Frobenius Error",
        title="Absolute Frobenius Error vs Time (Decimated)",
        out_png=os.path.join(NORM_PLOTS_DIR, "error_absolute_vs_time.png"),
    )

    plot_two_series(
        t=t_dec,
        y1=err_disc_rel_t,
        y2=err_cont_rel_t,
        label1="||E_disc||_F / ||Y||_F",
        label2="||E_cont||_F / ||Y||_F",
        ylabel="Relative Frobenius Error",
        title="Relative Frobenius Error vs Time (Decimated)",
        out_png=os.path.join(NORM_PLOTS_DIR, "error_relative_vs_time.png"),
    )

    plot_two_series(
        t=t_dec,
        y1=err_disc_norm0_t,
        y2=err_cont_norm0_t,
        label1="||E_disc||_F / ||Y(0)||_F",
        label2="||E_cont||_F / ||Y(0)||_F",
        ylabel="Error normalized by ||Y(0)||_F",
        title="Error Normalized to Initial Frobenius Norm vs Time (Decimated)",
        out_png=os.path.join(NORM_PLOTS_DIR, "error_norm0_vs_time.png"),
    )

    print(f"[OK] Saved error plots -> {NORM_PLOTS_DIR}/")

    # 3) Snapshot plots (frequency-based) → refreshed "dmd_snapshots"
    if SNAPSHOT_FREQ > 0:
        refresh_dir(SNAPSHOT_DIR)
        make_snapshot_plots(
            Y_truth=Y_truth,
            Y_disc=Yhat_disc,
            Y_cont=Yhat_cont,
            t=t_dec,
            freq=SNAPSHOT_FREQ,
            out_dir=SNAPSHOT_DIR,
        )
        print(f"\n[OK] Saved snapshot figures in -> {SNAPSHOT_DIR}/")
    else:
        print("[INFO] SNAPSHOT_FREQ == 0, skipping snapshot plots.")


# ----------------------------
# Plotting helpers
# ----------------------------
def plot_two_series(t, y1, y2, label1, label2, ylabel, title, out_png):
    """Generic helper for 2-line time series plots."""
    plt.figure(figsize=(8, 5))
    plt.plot(t, y1, lw=2, label=label1)
    plt.plot(t, y2, lw=2, ls="--", label=label2)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def make_snapshot_plots(Y_truth, Y_disc, Y_cont, t, freq, out_dir):
    """
    For each k in [0, t.size-1] with k % freq == 0, plot the three fields over space:
    - True (solid)
    - Discrete DMD (dashed)
    - Continuous DMD (dotted)
    Save each as a PNG in out_dir.
    Shows progress inline: "Plotting snapshot X / N".
    """
    if Y_truth.shape != Y_disc.shape or Y_truth.shape != Y_cont.shape:
        raise ValueError("Truth, discrete, and continuous arrays must have the same shape.")

    n_x, n_time = Y_truth.shape
    x = np.arange(n_x)
    indices = [k for k in range(n_time) if k % freq == 0]
    total = len(indices)

    for i, k in enumerate(indices, 1):
        # progress counter (single-line update)
        print(f"\r[Progress] Plotting snapshot {i}/{total} (k={k})", end="")
        sys.stdout.flush()

        plt.figure(figsize=(8, 5))
        plt.plot(x, Y_truth[:, k], **STYLE_TRUE)
        plt.plot(x, Y_disc[:, k],  **STYLE_DISC)
        plt.plot(x, Y_cont[:, k],  **STYLE_CONT)
        plt.xlabel("Spatial index")
        plt.ylabel("Field value")
        plt.title(f"Snapshot k={k}, t={t[k]:.4g}")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        fname = os.path.join(out_dir, f"snapshot_k{k:05d}.png")
        plt.savefig(fname, dpi=150)
        plt.close()

    # finish progress line cleanly
    print()


# ----------------------------
# Filesystem helper
# ----------------------------
def refresh_dir(path):
    """Create a fresh directory (delete if exists, then recreate)."""
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


# ----------------------------
# Entrypoint
# ----------------------------
if __name__ == "__main__":
    main()


