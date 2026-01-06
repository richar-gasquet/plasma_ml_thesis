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
NPZ_PATH = "dmd_density2_scenarios_ANALYSIS.npz"

# Output directories (mirroring structure of diffusion_bounded_linear/dmd_plots.py)
BASE_DIR = "dmd_density2_scenarios_plots"
NORM_PLOTS_DIR = os.path.join(BASE_DIR, "dmd_norm_plots")   # refreshed each run
SNAPSHOT_DIR   = os.path.join(BASE_DIR, "dmd_snapshots")    # refreshed each run

# Snapshot plotting frequency:
#   0 -> plot none
#   k -> plot every k-th snapshot (k >= 1)
SNAPSHOT_FREQ = 0  # Disabled for multi-scenario comparison

# Optional: line styles for snapshot overlays
STYLE_TRUE = dict(lw=2, label="Truth")
STYLE_DISC = dict(lw=2, ls="--", label="DMD (discrete)")
STYLE_CONT = dict(lw=2, ls=":",  label="DMD (continuous)")


def main():
    # 1) Load NPZ from Script 1
    if not os.path.exists(NPZ_PATH):
        raise FileNotFoundError(f"Could not find NPZ file: {NPZ_PATH}")

    data = np.load(NPZ_PATH, allow_pickle=True)

    # Check if this is a multi-scenario file
    num_scenarios = data.get("num_scenarios", 1)
    
    if num_scenarios > 1:
        # Multi-scenario comparison
        scenarios = data["scenarios"]
        print(f"Loading {num_scenarios} scenarios: {scenarios}")
        
        # 2) Create comparison plot with all scenarios
        refresh_dir(NORM_PLOTS_DIR)
        
        plot_multi_scenario_comparison(data, num_scenarios, NORM_PLOTS_DIR)
        
        print(f"[OK] Saved multi-scenario comparison plot -> {NORM_PLOTS_DIR}/")
    else:
        # Original single-scenario workflow (for backward compatibility)
        t_dec = data["t_decimated"]
        err_disc_abs_t = data["err_disc_abs_t"]
        err_cont_abs_t = data["err_cont_abs_t"]
        err_disc_rel_t = data["err_disc_rel_t"]
        err_cont_rel_t = data["err_cont_rel_t"]
        err_disc_norm0_t = data["err_disc_norm0_t"]
        err_cont_norm0_t = data["err_cont_norm0_t"]
        
        Y_truth = data.get("Y_truth", None)
        Yhat_disc = data.get("Yhat_disc", None)
        Yhat_cont = data.get("Yhat_cont", None)
        
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
        
        if SNAPSHOT_FREQ > 0 and Y_truth is not None:
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


def plot_multi_scenario_comparison(data, num_scenarios, out_dir):
    """
    Create a single comparison plot showing Frobenius error norms for all scenarios.
    """
    # Color and style options for different scenarios
    # Use a colormap that works well for many scenarios
    num_scenarios = int(num_scenarios)  # Ensure it's a Python int
    if num_scenarios <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, num_scenarios))
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, num_scenarios))
    
    # Create figure with subplots for different error metrics
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot relative error (most informative)
    ax1 = axes[0]
    # Plot absolute error
    ax2 = axes[1]
    # Plot normalized-to-initial error
    ax3 = axes[2]
    
    # Collect all error values to determine reasonable bounds
    all_err_rel = []
    all_err_abs = []
    all_err_norm0 = []
    
    for scenario_idx in range(num_scenarios):
        scenario_key = f"scenario_{scenario_idx}"
        err_disc_rel = data[f"{scenario_key}_err_disc_rel_t"]
        err_disc_abs = data[f"{scenario_key}_err_disc_abs_t"]
        err_disc_norm0 = data[f"{scenario_key}_err_disc_norm0_t"]
        all_err_rel.append(err_disc_rel)
        all_err_abs.append(err_disc_abs)
        all_err_norm0.append(err_disc_norm0)
    
    # Compute percentiles to set reasonable y-axis limits (use 95th percentile to exclude extreme outliers)
    err_rel_95 = np.percentile(np.concatenate(all_err_rel), 95)
    err_abs_95 = np.percentile(np.concatenate(all_err_abs), 95)
    err_norm0_95 = np.percentile(np.concatenate(all_err_norm0), 95)
    
    for scenario_idx in range(num_scenarios):
        scenario_key = f"scenario_{scenario_idx}"
        N = int(data[f"{scenario_key}_N"])
        M = int(data[f"{scenario_key}_M"])
        t = data[f"{scenario_key}_t_decimated"]
        err_disc_rel = data[f"{scenario_key}_err_disc_rel_t"]
        err_cont_rel = data[f"{scenario_key}_err_cont_rel_t"]
        err_disc_abs = data[f"{scenario_key}_err_disc_abs_t"]
        err_cont_abs = data[f"{scenario_key}_err_cont_abs_t"]
        err_disc_norm0 = data[f"{scenario_key}_err_disc_norm0_t"]
        err_cont_norm0 = data[f"{scenario_key}_err_cont_norm0_t"]
        
        label = f"N={N}"
        color = colors[scenario_idx]
        ls = '-'  # Use solid lines for all, differentiate by color
        
        # Clip errors to reasonable bounds for visualization
        err_disc_rel_clipped = np.clip(err_disc_rel, 0, err_rel_95 * 1.1)
        err_disc_abs_clipped = np.clip(err_disc_abs, 0, err_abs_95 * 1.1)
        err_disc_norm0_clipped = np.clip(err_disc_norm0, 0, err_norm0_95 * 1.1)
        
        # Relative error - use discrete-time (more standard)
        ax1.plot(t, err_disc_rel_clipped, lw=2.5, ls=ls, color=color, label=label, alpha=0.9)
        
        # Absolute error
        ax2.plot(t, err_disc_abs_clipped, lw=2.5, ls=ls, color=color, label=label, alpha=0.9)
        
        # Normalized-to-initial error
        ax3.plot(t, err_disc_norm0_clipped, lw=2.5, ls=ls, color=color, label=label, alpha=0.9)
        
        # Add vertical line at training/forecast boundary
        t_train_end = t[N] if N < len(t) else t[-1]
        ax1.axvline(x=t_train_end, color=color, linestyle=':', alpha=0.4, linewidth=1.5)
        ax2.axvline(x=t_train_end, color=color, linestyle=':', alpha=0.4, linewidth=1.5)
        ax3.axvline(x=t_train_end, color=color, linestyle=':', alpha=0.4, linewidth=1.5)
    
    # Configure subplots
    # Adjust legend columns based on number of scenarios
    ncol_legend = min(4, max(2, (num_scenarios + 2) // 3))
    legend_fontsize = max(7, min(9, 11 - num_scenarios // 6))
    
    # Set y-axis limits based on 95th percentile (with small margin)
    ax1.set_ylim(bottom=0, top=err_rel_95 * 1.15)
    ax2.set_ylim(bottom=0, top=err_abs_95 * 1.15)
    ax3.set_ylim(bottom=0, top=err_norm0_95 * 1.15)
    
    ax1.set_ylabel("Relative Frobenius Error\n||E||_F / ||Y||_F", fontsize=11)
    ax1.set_title("Multi-Scenario Comparison: Relative Frobenius Error vs Time (clipped at 95th percentile)", fontsize=13, fontweight='bold')
    ax1.legend(ncol=ncol_legend, fontsize=legend_fontsize, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("Time", fontsize=10)
    
    ax2.set_ylabel("Absolute Frobenius Error\n||E||_F", fontsize=11)
    ax2.set_title("Multi-Scenario Comparison: Absolute Frobenius Error vs Time (clipped at 95th percentile)", fontsize=13, fontweight='bold')
    ax2.legend(ncol=ncol_legend, fontsize=legend_fontsize, loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("Time", fontsize=10)
    
    ax3.set_ylabel("Error Normalized to Initial\n||E||_F / ||Y(0)||_F", fontsize=11)
    ax3.set_title("Multi-Scenario Comparison: Normalized-to-Initial Error vs Time (clipped at 95th percentile)", fontsize=13, fontweight='bold')
    ax3.legend(ncol=ncol_legend, fontsize=legend_fontsize, loc='best', framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel("Time", fontsize=10)
    
    plt.tight_layout()
    
    out_png = os.path.join(out_dir, "multi_scenario_comparison.png")
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {out_png}")


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


