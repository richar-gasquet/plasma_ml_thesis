#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 2 — Post-processing & plots for DMD analysis.

What it does:
1) Loads the NPZ file produced by Script 1.
2) Plots time series of:
   - Absolute Frobenius error: ||Yhat - Y||_F
   - Relative Frobenius error: ||Yhat - Y||_F / ||Y||_F
   - Normalized-to-initial error: ||Yhat - Y||_F / ||Y(0)||_F
   All three plots are saved into a refreshed directory "dmd_norm_plots".
3) Plots selected snapshots (original noisy, smoothed, DMD discrete, DMD continuous) 
   at a user-specified frequency; saved in a refreshed directory "dmd_snapshots".
"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import sys

# ----------------------------
# User config
# ----------------------------
NPZ_PATH = "dmd_rollout_outputs_ANALYSIS.npz"

# Output directories
BASE_DIR = "dmd_hankel_plots"
NORM_PLOTS_DIR = os.path.join(BASE_DIR, "dmd_norm_plots")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "dmd_snapshots")

# Snapshot plotting frequency:
#   0 -> plot none
#   k -> plot every k-th snapshot (k >= 1)
SNAPSHOT_FREQ = 50

# Optional: line styles for snapshot overlays
STYLE_NOISY = dict(lw=1.5, alpha=0.6, label="Original noisy")
STYLE_SMOOTH = dict(lw=2, label="Smoothed")
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
    # Errors vs smoothed data
    err_disc_abs_smooth_t = data["err_disc_abs_smooth_t"]
    err_cont_abs_smooth_t = data["err_cont_abs_smooth_t"]
    err_disc_rel_smooth_t = data["err_disc_rel_smooth_t"]
    err_cont_rel_smooth_t = data["err_cont_rel_smooth_t"]
    err_disc_norm0_smooth_t = data["err_disc_norm0_smooth_t"]
    err_cont_norm0_smooth_t = data["err_cont_norm0_smooth_t"]
    
    # Errors vs noisy data
    err_disc_abs_noisy_t = data["err_disc_abs_noisy_t"]
    err_cont_abs_noisy_t = data["err_cont_abs_noisy_t"]
    err_disc_rel_noisy_t = data["err_disc_rel_noisy_t"]
    err_cont_rel_noisy_t = data["err_cont_rel_noisy_t"]
    err_disc_norm0_noisy_t = data["err_disc_norm0_noisy_t"]
    err_cont_norm0_noisy_t = data["err_cont_norm0_noisy_t"]

    # Fields for snapshot comparisons (aligned to t_dec)
    Y_noisy   = data["Y_noisy"]
    Y_truth   = data["Y_truth"]
    Yhat_disc = data["Yhat_disc"]
    Yhat_cont = data["Yhat_cont"]

    # 2) Error plots → refreshed "dmd_norm_plots"
    refresh_dir(NORM_PLOTS_DIR)

    plot_four_series(
        t=t_dec,
        y1=err_disc_abs_noisy_t, y2=err_cont_abs_noisy_t,
        y3=err_disc_abs_smooth_t, y4=err_cont_abs_smooth_t,
        label1="DMD disc vs noisy", label2="DMD cont vs noisy",
        label3="DMD disc vs smooth", label4="DMD cont vs smooth",
        ylabel="Absolute Frobenius Error",
        title="Absolute Frobenius Error vs Time (Decimated)",
        out_png=os.path.join(NORM_PLOTS_DIR, "error_absolute_vs_time.png")
    )

    plot_four_series(
        t=t_dec,
        y1=err_disc_rel_noisy_t, y2=err_cont_rel_noisy_t,
        y3=err_disc_rel_smooth_t, y4=err_cont_rel_smooth_t,
        label1="DMD disc vs noisy", label2="DMD cont vs noisy",
        label3="DMD disc vs smooth", label4="DMD cont vs smooth",
        ylabel="Relative Frobenius Error",
        title="Relative Frobenius Error vs Time (Decimated)",
        out_png=os.path.join(NORM_PLOTS_DIR, "error_relative_vs_time.png")
    )

    plot_four_series(
        t=t_dec,
        y1=err_disc_norm0_noisy_t, y2=err_cont_norm0_noisy_t,
        y3=err_disc_norm0_smooth_t, y4=err_cont_norm0_smooth_t,
        label1="DMD disc vs noisy", label2="DMD cont vs noisy",
        label3="DMD disc vs smooth", label4="DMD cont vs smooth",
        ylabel="Error normalized by ||Y(0)||_F",
        title="Error Normalized to Initial Frobenius Norm vs Time (Decimated)",
        out_png=os.path.join(NORM_PLOTS_DIR, "error_norm0_vs_time.png")
    )

    print(f"[OK] Saved error plots -> {NORM_PLOTS_DIR}/")

    # 3) Snapshot plots (frequency-based) → refreshed "dmd_snapshots"
    if SNAPSHOT_FREQ > 0:
        refresh_dir(SNAPSHOT_DIR)
        make_snapshot_plots(
            Y_noisy=Y_noisy,
            Y_smooth=Y_truth,
            Y_disc=Yhat_disc,
            Y_cont=Yhat_cont,
            t=t_dec,
            freq=SNAPSHOT_FREQ,
            out_dir=SNAPSHOT_DIR
        )
        print(f"\n[OK] Saved snapshot figures in -> {SNAPSHOT_DIR}/")
    else:
        print("[INFO] SNAPSHOT_FREQ == 0, skipping snapshot plots.")


# ----------------------------
# Plotting helpers
# ----------------------------
def plot_four_series(t, y1, y2, y3, y4, label1, label2, label3, label4, ylabel, title, out_png):
    """Generic helper for 4-line time series plots."""
    plt.figure(figsize=(10, 6))
    plt.plot(t, y1, lw=2, ls="-", alpha=0.7, label=label1)
    plt.plot(t, y2, lw=2, ls="--", alpha=0.7, label=label2)
    plt.plot(t, y3, lw=2, ls="-", alpha=0.9, label=label3)
    plt.plot(t, y4, lw=2, ls=":", alpha=0.9, label=label4)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def make_snapshot_plots(Y_noisy, Y_smooth, Y_disc, Y_cont, t, freq, out_dir):
    """
    For each k in [0, t.size-1] with k % freq == 0, plot the four fields over space:
    - Original noisy data (thin, semi-transparent)
    - Smoothed data (solid)
    - Discrete DMD (dashed)
    - Continuous DMD (dotted)
    Save each as a PNG in out_dir.
    Shows progress inline: "Plotting snapshot X / N".
    """
    if Y_noisy.shape != Y_smooth.shape or Y_smooth.shape != Y_disc.shape or Y_smooth.shape != Y_cont.shape:
        raise ValueError("All arrays (noisy, smooth, discrete, continuous) must have the same shape.")

    n_x, n_time = Y_smooth.shape
    x = np.arange(n_x)
    indices = [k for k in range(n_time) if k % freq == 0]
    total = len(indices)

    for i, k in enumerate(indices, 1):
        print(f"\r[Progress] Plotting snapshot {i}/{total} (k={k})", end="")
        sys.stdout.flush()

        plt.figure(figsize=(8, 5))
        plt.plot(x, Y_noisy[:, k], **STYLE_NOISY)
        plt.plot(x, Y_smooth[:, k], **STYLE_SMOOTH)
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

