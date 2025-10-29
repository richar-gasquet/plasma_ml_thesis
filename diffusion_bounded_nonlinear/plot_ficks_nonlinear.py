#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

# -------------------------------------------
# User options
# -------------------------------------------
OUTDIR = "frames"       # single output dir (smooth data only)
DPI = 130
NPRINT = 100           # save every N-th frame (set to 1 to save all)
SHOW_TITLE_DT = True    # include dt in title if available
LINEWIDTH = 2.0

# --- MUST MATCH THE SOLVER ---
nu_min = 1e-4
nu0    = 1e-2
# -----------------------------

# Nonlinear term styling (separate subplot)
NL_LABEL = r"$\partial_x(D\,u_x)$"
NL_LINEWIDTH = 1.5
NL_LINESTYLE = "--"

TOTAL_DENSITY_PNG = "total_density_vs_time.png"

def main():
    # --- Load data (smooth only) ---
    x   = np.loadtxt("x.txt")                 # (N,)
    U   = np.loadtxt("U.txt")                 # (N, T)
    dt  = _read_dt_file("dt.txt")

    # basic checks
    if U.shape[0] != x.shape[0]:
        raise ValueError("Grid length mismatch: x vs U")

    # --- Compute NL internally using the provided stencil ---
    dx = float(x[1] - x[0])  # uniform grid assumed
    NL = np.zeros_like(U)
    for j in range(U.shape[1]):
        NL[:, j] = nonlinear_term(U[:, j], dx)

    # --- Plot frames with stacked subplots ---
    make_frames(
        x,
        U=U,
        NL=NL,
        label="u(x, t)",
        outdir=OUTDIR,
        dt=dt,
        dpi=DPI,
        nprint=NPRINT,
        lw=LINEWIDTH
    )

    # --- Total density vs time (from smooth U) ---
    t = np.arange(U.shape[1]) * dt if dt is not None else np.arange(U.shape[1])
    total_density = _integrate_over_x(x, U)
    _plot_total_density(t, total_density, dt, TOTAL_DENSITY_PNG)

    print(f"Done. Wrote frames to: {OUTDIR}/")
    print(f"Wrote total-density plot: {TOTAL_DENSITY_PNG}")

def make_frames(
    x,
    U,
    NL,
    label,
    outdir,
    dt=None,
    dpi=120,
    nprint=1,
    lw=2.0
):
    """
    Save frames for the smooth dataset at each time snapshot, with
    the nonlinear diffusion term in a second (stacked) subplot.

    U:  array (N, T) of the solution
    NL: array (N, T) of ∂x(D u_x)
    """
    outdir = Path(outdir)
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # stable y-limits per panel across all frames
    y_min_u, y_max_u = U.min(), U.max()
    pad_u = 0.05 * (y_max_u - y_min_u if y_max_u > y_min_u else 1.0)
    y_min_u -= pad_u; y_max_u += pad_u

    y_min_nl, y_max_nl = NL.min(), NL.max()
    pad_nl = 0.05 * (y_max_nl - y_min_nl if y_max_nl > y_min_nl else 1.0)
    y_min_nl -= pad_nl; y_max_nl += pad_nl

    N, T = U.shape
    if NL.shape != (N, T):
        raise ValueError("Shape mismatch between U and NL.")

    # which frames to save
    indices = list(range(0, T, max(1, nprint)))

    for count, j in enumerate(indices, start=1):
        print(f"[{outdir.name}] saving frame {count}/{len(indices)} (t-index={j})")

        fig, (ax_u, ax_nl) = plt.subplots(
            2, 1, figsize=(7.0, 6.2), sharex=True,
            gridspec_kw={'height_ratios': [2, 1]}
        )

        # --- Top: solution u(x,t) ---
        ax_u.plot(x, U[:, j], lw=lw, label=label)
        ax_u.set_xlim(x[0], x[-1])
        ax_u.set_ylim(y_min_u, y_max_u)
        ax_u.set_ylabel("u(x, t)")
        ax_u.legend(loc="best", frameon=True)

        # --- Bottom: nonlinear term ∂x(D u_x) ---
        ax_nl.plot(x, NL[:, j], NL_LINESTYLE, lw=NL_LINEWIDTH, label=NL_LABEL)
        ax_nl.set_ylim(y_min_nl, y_max_nl)
        ax_nl.set_xlabel("x")
        ax_nl.set_ylabel(NL_LABEL)
        ax_nl.legend(loc="best", frameon=True)

        # title
        if SHOW_TITLE_DT and (dt is not None):
            fig.suptitle(f"t = {j*dt:.6f}   (dt = {dt:.3e})")
        else:
            fig.suptitle(f"Snapshot j = {j}")

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

# ---------- helpers to compute NL internally ----------
def face_gradients(u, dx):
    """g_{i+1/2} = (u_{i+1} - u_i)/dx for i=0..N-2 (size N-1)."""
    return (u[1:] - u[:-1]) / dx

def nonlinear_term(u, dx):
    """
    Return ∂x(D u_x) evaluated with the same discrete operators
    used inside the RHS (so it matches what the solver advances).
    """
    g = face_gradients(u, dx)
    # face-centered u and D(u)=nu_min+nu0*u^2
    u_faces = 0.5 * (u[1:] + u[:-1])                         # <<< CHANGED
    D = nu_min + nu0 * (u_faces**2)                          # <<< CHANGED
    F = -D * g
    NL = np.zeros_like(u)
    NL[1:-1] = - (F[1:] - F[:-1]) / dx
    NL[0] = 0.0
    NL[-1] = 0.0
    return NL

# ------------------------------------------------------

if __name__ == "__main__":
    main()
