#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

# -------------------------------------------
# User options
# -------------------------------------------
OUTDIR_CLEAN = "frames_clean"
OUTDIR_NOISY = "frames_noisy"
DPI = 130
NPRINT = 500          # save every N-th frame (set to 1 to save all)
SHOW_TITLE_DT = True # include dt in title if available
LINEWIDTH = 2.0

# Steady-state overlay style
STEADY_LABEL = "Steady (analytic)"
STEADY_LINEWIDTH = 1.0
STEADY_LINESTYLE = "--"
STEADY_COLOR = "k"

TOTAL_DENSITY_PNG = "total_density_vs_time.png"

def main():
    # --- Load data (diffusion outputs) ---
    x    = np.loadtxt("x.txt")
    U    = np.loadtxt("U.txt")            # (N, T)
    U_n  = np.loadtxt("U_noisy.txt")      # (N, T)
    U_ss = np.loadtxt("U_steady.txt")     # (N,)

    # basic checks
    if U.shape[0] != x.shape[0]:
        raise ValueError("Grid length mismatch: x vs U")
    if U_n.shape != U.shape:
        raise ValueError("Shape mismatch between U and U_noisy.")
    if U_ss.shape[0] != x.shape[0]:
        raise ValueError("Grid length mismatch: x vs U_steady")

    # read dt
    dt = _read_dt_file("dt.txt")

    # --- Plot CLEAN frames (with steady overlay) ---
    make_frames(
        x,
        U=U,
        label="Diffusion (clean)",
        U_steady=U_ss,
        outdir=OUTDIR_CLEAN,
        dt=dt,
        dpi=DPI,
        nprint=NPRINT,
        lw=LINEWIDTH
    )

    # --- Plot NOISY frames (with steady overlay) ---
    make_frames(
        x,
        U=U_n,
        label="Diffusion (noisy)",
        U_steady=U_ss,
        outdir=OUTDIR_NOISY,
        dt=dt,
        dpi=DPI,
        nprint=NPRINT,
        lw=LINEWIDTH
    )

    # --- Total density vs time (clean) ---
    if dt is None:
        t = np.arange(U.shape[1])
    else:
        t = np.arange(U.shape[1]) * dt
    total_density = _integrate_over_x(x, U)
    _plot_total_density(t, total_density, dt, TOTAL_DENSITY_PNG)

    print(f"Done. Wrote frames to: {OUTDIR_CLEAN}/ and {OUTDIR_NOISY}/")
    print(f"Wrote total-density plot: {TOTAL_DENSITY_PNG}")

def make_frames(
    x,
    U,
    label,
    U_steady,
    outdir,
    dt=None,
    dpi=120,
    nprint=1,
    lw=2.0
):
    """
    Save frames for a single dataset at each time snapshot, with steady-state overlay.

    U:        array (N, T)
    U_steady: array (N,)
    """
    outdir = Path(outdir)
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # y-limits across all snapshots including steady profile
    y_min = min(U.min(), U_steady.min())
    y_max = max(U.max(), U_steady.max())
    y_pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    y_min -= y_pad
    y_max += y_pad

    N, T = U.shape
    if U_steady.shape[0] != N:
        raise ValueError("Shape mismatch between U and U_steady (N dimension).")

    # choose which frames to save
    indices = list(range(0, T, max(1, nprint)))

    for count, j in enumerate(indices, start=1):
        print(f"[{outdir.name}] saving frame {count}/{len(indices)} (t-index={j})")

        fig, ax = plt.subplots(figsize=(6.8, 3.6))
        # time slice
        ax.plot(x, U[:, j], lw=lw, label=label)
        # steady overlay (thin black dashed)
        ax.plot(
            x, U_steady,
            linestyle=STEADY_LINESTYLE,
            lw=STEADY_LINEWIDTH,
            color=STEADY_COLOR,
            label=STEADY_LABEL
        )

        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("x")
        ax.set_ylabel("u(x, t)")

        if SHOW_TITLE_DT and (dt is not None):
            ax.set_title(f"t = {j*dt:.6f}   (dt = {dt:.3e})")
        else:
            ax.set_title(f"Snapshot j = {j}")

        ax.legend(loc="best", frameon=True)
        fig.tight_layout()
        fig.savefig(outdir / f"{j:04d}.png", dpi=dpi)
        plt.close(fig)

def _read_dt_file(path):
    p = Path(path)
    if not p.exists():
        print(f"[info] {path} not found; titles will omit dt.")
        return None
    try:
        txt = p.read_text().strip().splitlines()
        return float(txt[0])
    except Exception as e:
        print(f"[warn] Failed to read {path}: {e}")
        return None

def _integrate_over_x(x, U):
    """Return total density (mass) vs time using trapezoidal integration over x."""
    return np.trapz(U, x=x, axis=0)

def _plot_total_density(t, M, dt, outfile):
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.plot(t, M, lw=2.0)
    ax.set_xlabel("time" if dt is not None else "snapshot index")
    ax.set_ylabel("Total density in domain")
    ax.set_title("Total density vs time" + (f"  (dt = {dt:.3e})" if dt is not None else ""))
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    main()
