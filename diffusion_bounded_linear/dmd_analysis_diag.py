#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 1 (analysis only): PyDMD-based affine (bias-augmented) DMD with
- Zero Dirichlet BCs (train on interior, re-pad zeros),
- Temporal decimation,
- Manual or energy-based rank selection,
- Discrete- and continuous-time rollouts,
- NO plotting (data saved for a separate plotting script).

Saves sufficient data for:
1) Estimated vs true solutions (fields over time),
2) Long-time estimated solutions from DMD (beyond truth),
3) Relative Frobenius error time-series (and absolute + normalized-to-initial variants).
"""

import os
import numpy as np
from pydmd import DMD, BOPDMD
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter


# ==========================================================
# Main
# ==========================================================
def main():
    # ---------- User config ----------
    input_path   = "U_noisy.txt"
    dt           = 1.0
    decimate_q   = 10
    N            = 100
    M            = 500

    rank_mode    = "manual"     # "manual" or "energy"
    r_manual     = 15
    energy_thr   = 0.999

    # Smoothing parameters
    temporal_sigma = 2.0  # Gaussian temporal smoothing (sigma in time steps)
    spatial_window = 20   # Spatial Savitzky-Golay window
    spatial_polyorder = 3  # Spatial Savitzky-Golay polynomial order

    save_npz     = "dmd_rollout_outputs_pydmd_refactored_ANALYSIS.npz"
    # ---------------------------------

    # 1) Load data
    X_full_noisy = load_snapshots(input_path).astype(float)  # Original noisy data

    # 2) Apply temporal Gaussian smoothing (along time axis for each spatial point)
    # Uses 'nearest' mode at boundaries to handle early times with only future data
    X_full_time_smooth = np.stack([
        gaussian_filter1d(X_full_noisy[i, :], sigma=temporal_sigma, mode='nearest')
        for i in range(X_full_noisy.shape[0])
    ], axis=0)

    # 3) Apply spatial smoothing (Savitzky-Golay) to noisy data
    # Smooth data spatially
    # X_full_smooth = np.stack([gaussian_filter1d(X_full_time_smooth[:, j], sigma=5.0) for j in range(X_full_time_smooth.shape[1])], axis=1)
    # X_full_smooth = np.stack([uniform_filter1d(X_full_time_smooth[:, j], size=10) for j in range(X_full_time_smooth.shape[1])], axis=1)
    X_full_smooth = np.stack([savgol_filter(X_full_time_smooth[:, j], window_length=spatial_window, polyorder=spatial_polyorder) for j in range(X_full_time_smooth.shape[1])], axis=1)

    # X_full_smooth = np.stack([gaussian_filter1d(X_full_noisy[:, j], sigma=5.0) for j in range(X_full_noisy.shape[1])], axis=1)
    # X_full_smooth = np.stack([uniform_filter1d(X_full_noisy[:, j], size=10) for j in range(X_full_noisy.shape[1])], axis=1)
    # X_full_smooth = np.stack([savgol_filter(X_full_noisy[:, j], window_length=spatial_window, polyorder=spatial_polyorder) for j in range(X_full_noisy.shape[1])], axis=1)

    # 4) Decimate (all versions: noisy and fully smoothed)
    X_full_noisy_dec, dt_eff = decimate_series(X_full_noisy, dt, decimate_q)
    X_full_time_smooth_dec, _ = decimate_series(X_full_time_smooth, dt, decimate_q)
    X_full, dt_eff = decimate_series(X_full_smooth, dt, decimate_q)
    n_x, n_time = X_full.shape
    validate_train_rollout(N, M, n_time)    


    # 2) Interior extraction (Dirichlet) + affine augmentation
    #print(X_full.shape)
    
    # time and spatial smoothed data
    # X_aug, n_int = build_interior_affine_augmented(X_full)

    # only time smoothed data
    X_aug, n_int = build_interior_affine_augmented(X_full_time_smooth_dec)
    #print(X_aug.shape)


    # 3) Rank selection
    X_train = X_aug[:, :N+1]
    svd_rank, energy_captured = choose_svd_rank(X_train, rank_mode, r_manual, energy_thr)


    # 4) Fit DMD
    dmd, Phi, evals, b0 = fit_dmd(X_train, svd_rank)
    # dmd, Phi, evals, b0 = fit_bopdmd(X_train, svd_rank, dt_eff)


    # 5) Rollouts
    X_hat_disc_full, X_hat_cont_full = rollout_full_field_full_length(
        Phi, evals, b0, n_x, n_int, N, M, dt_eff
    )

    # Truncate to available truth for comparisons
    t_max   = min(N + M, n_time - 1)
    Y_truth = X_full[:, :t_max+1]  # Fully smoothed data (decimated)
    Y_noisy = X_full_noisy_dec[:, :t_max+1]  # Original noisy data (decimated)
    Y_time_smooth = X_full_time_smooth_dec[:, :t_max+1]  # Time-smoothed only (decimated)
    Yhat_d  = X_hat_disc_full[:, :t_max+1]
    Yhat_c  = X_hat_cont_full[:, :t_max+1]

    # 6) Relative/absolute error Frobenius norms (per-time)
    # Errors vs smoothed data
    fro_truth_t = np.linalg.norm(Y_truth, axis=0)                  # ||Y_smooth(t)||_F
    err_disc_abs_smooth_t = np.linalg.norm(Yhat_d - Y_truth, axis=0)      # ||E_disc(t)||_F vs smooth
    err_cont_abs_smooth_t = np.linalg.norm(Yhat_c - Y_truth, axis=0)      # ||E_cont(t)||_F vs smooth

    # Relative (divide by ||Y_smooth(t)||_F); safe divide
    err_disc_rel_smooth_t = np.divide(err_disc_abs_smooth_t, fro_truth_t,
                                       out=np.zeros_like(err_disc_abs_smooth_t), where=fro_truth_t > 0)
    err_cont_rel_smooth_t = np.divide(err_cont_abs_smooth_t, fro_truth_t,
                                       out=np.zeros_like(err_cont_abs_smooth_t), where=fro_truth_t > 0)

    # Normalized-to-initial (divide by ||Y_smooth(0)||_F)
    fro0_smooth = fro_truth_t[0] if fro_truth_t.size and fro_truth_t[0] != 0 else 1.0
    err_disc_norm0_smooth_t = err_disc_abs_smooth_t / fro0_smooth
    err_cont_norm0_smooth_t = err_cont_abs_smooth_t / fro0_smooth

    # Errors vs noisy data
    fro_noisy_t = np.linalg.norm(Y_noisy, axis=0)                   # ||Y_noisy(t)||_F
    err_disc_abs_noisy_t = np.linalg.norm(Yhat_d - Y_noisy, axis=0)      # ||E_disc(t)||_F vs noisy
    err_cont_abs_noisy_t = np.linalg.norm(Yhat_c - Y_noisy, axis=0)      # ||E_cont(t)||_F vs noisy

    # Relative (divide by ||Y_noisy(t)||_F); safe divide
    err_disc_rel_noisy_t = np.divide(err_disc_abs_noisy_t, fro_noisy_t,
                                     out=np.zeros_like(err_disc_abs_noisy_t), where=fro_noisy_t > 0)
    err_cont_rel_noisy_t = np.divide(err_cont_abs_noisy_t, fro_noisy_t,
                                     out=np.zeros_like(err_cont_abs_noisy_t), where=fro_noisy_t > 0)

    # Normalized-to-initial (divide by ||Y_noisy(0)||_F)
    fro0_noisy = fro_noisy_t[0] if fro_noisy_t.size and fro_noisy_t[0] != 0 else 1.0
    err_disc_norm0_noisy_t = err_disc_abs_noisy_t / fro0_noisy
    err_cont_norm0_noisy_t = err_cont_abs_noisy_t / fro0_noisy

    # 7) Save all required data
    np.savez_compressed(
        save_npz,
        # Raw inputs & configuration
        X_full_noisy=X_full_noisy,  # Original noisy data (before smoothing, before decimation)
        X_full_time_smooth=X_full_time_smooth,  # Time-smoothed only (before decimation)
        X_full_smooth=X_full_smooth,  # Fully smoothed (spatial only, before decimation)
        X_full_decimated=X_full,  # Fully smoothed data (after decimation)
        # temporal_sigma=temporal_sigma,  # Temporal smoothing parameter
        dt=dt, dt_eff=dt_eff, decimate_q=decimate_q,
        N=N, M=M, rank_mode=rank_mode, svd_rank=svd_rank,
        energy_captured=energy_captured,
        # DMD factors
        modes=Phi, evals=evals, amplitudes=b0,
        singular_values=np.array(getattr(dmd, 'singular_values', [])),
        # Predictions (full field, full-length rollout 0..N+M)
        X_hat_disc_full=X_hat_disc_full,
        X_hat_cont_full=X_hat_cont_full,
        # Truth & truncated predictions for aligned comparisons (0..t_max)
        Y_truth=Y_truth, Y_noisy=Y_noisy, # Y_time_smooth=Y_time_smooth,
        Yhat_disc=Yhat_d, Yhat_cont=Yhat_c,
        # Time axes
        t_decimated=np.arange(t_max + 1) * dt_eff,
        t_full_prediction_window=np.arange(N + M + 1) * dt_eff,
        t_max=t_max,
        # === Error Frobenius norms (time-resolved) ===
        # Errors vs smoothed data
        err_disc_abs_smooth_t=err_disc_abs_smooth_t,
        err_cont_abs_smooth_t=err_cont_abs_smooth_t,
        err_disc_rel_smooth_t=err_disc_rel_smooth_t,
        err_cont_rel_smooth_t=err_cont_rel_smooth_t,
        err_disc_norm0_smooth_t=err_disc_norm0_smooth_t,
        err_cont_norm0_smooth_t=err_cont_norm0_smooth_t,
        # Errors vs noisy data
        err_disc_abs_noisy_t=err_disc_abs_noisy_t,
        err_cont_abs_noisy_t=err_cont_abs_noisy_t,
        err_disc_rel_noisy_t=err_disc_rel_noisy_t,
        err_cont_rel_noisy_t=err_cont_rel_noisy_t,
        err_disc_norm0_noisy_t=err_disc_norm0_noisy_t,
        err_cont_norm0_noisy_t=err_cont_norm0_noisy_t,
    )

    # 8) Save early-time diagnostic snapshots (to verify temporal smoothing boundaries)
    # save_early_snapshots(X_full_noisy, X_full_time_smooth, X_full_smooth, 
    #                      save_dir="temporal_smoothing_diagnostics")

    # 9) Analyse DMD
    dmd_diagnostics(X_train, dt=None, svd_rank=svd_rank, tlsq_rank=0, exact=True)
        

    print(f"[OK] Analysis complete. Data saved to: {save_npz}")


# ==========================================================
# Helper functions (unchanged)
# ==========================================================
def load_snapshots(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        X = np.load(path)
    elif ext == ".npz":
        data = np.load(path)
        if "X" not in data:
            raise ValueError("For .npz input, store the snapshot matrix under key 'X'.")
        X = data["X"]
    elif ext in [".csv", ".txt"]:
        try:
            X = np.loadtxt(path, delimiter=",")
            if X.ndim != 2: raise ValueError
        except Exception:
            X = np.loadtxt(path)
    else:
        raise ValueError(f"Unsupported input format: {ext}")
    if X.ndim != 2:
        raise ValueError("Input snapshots must be 2D: (n_x, n_time).")
    if X.shape[0] < 3:
        raise ValueError("Need ≥3 spatial points for Dirichlet BCs.")
    return X


def decimate_series(X, dt, q):
    if q < 1 or int(q) != q:
        raise ValueError("decimate_q must be a positive integer.")
    return X[:, ::int(q)], dt * int(q)


def validate_train_rollout(N, M, n_time):
    if not (1 <= N < n_time):
        raise ValueError(f"N must be in [1, n_time-1]. Got N={N}, n_time={n_time}.")
    if M < 1:
        raise ValueError("M must be >= 1.")
    if N + M >= n_time:
        print(f"[WARN] Ground truth truncates rollout: N+M={N+M} >= n_time={n_time}.")


def build_interior_affine_augmented(X_full):
    X_int = X_full[1:-1, :]
    ones = np.ones((1, X_int.shape[1]))
    return np.vstack([X_int, ones]), X_int.shape[0]
    #return X_int, X_int.shape[0]


def choose_svd_rank(X_train, mode, r_manual, energy_thr):
    if mode == "manual":
        svd_rank = int(max(1, min(r_manual, min(X_train.shape))))
        _, S, _ = np.linalg.svd(X_train, full_matrices=False)
        return svd_rank, cumulative_energy(S, min(svd_rank, S.size))
    if mode == "energy":
        _, S, _ = np.linalg.svd(X_train, full_matrices=False)
        if S.size == 0 or np.allclose(S, 0):
            raise ValueError("No energy in training data.")
        cume = np.cumsum(S**2) / np.sum(S**2)
        r_energy = int(np.searchsorted(cume, energy_thr) + 1)
        svd_rank = max(1, r_energy)
        return svd_rank, cumulative_energy(S, min(svd_rank, S.size))
    raise ValueError("rank_mode must be 'manual' or 'energy'.")


def cumulative_energy(svals, r):
    r = max(0, min(r, svals.size))
    if r == 0: return 0.0
    e2 = svals**2
    den = e2.sum()
    return 0.0 if den == 0.0 else float(e2[:r].sum() / den)


def save_early_snapshots(X_noisy, X_time_smooth, X_full_smooth, save_dir="temporal_smoothing_diagnostics", n_snapshots=5):
    """
    Save early-time snapshots to verify temporal smoothing boundary handling.
    Plots noisy, time-smoothed, and fully-smoothed data at early time steps.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    n_x = X_noisy.shape[0]
    x = np.arange(n_x)
    
    for k in range(min(n_snapshots, X_noisy.shape[1])):
        plt.figure(figsize=(10, 6))
        plt.plot(x, X_noisy[:, k], 'o-', lw=1.5, alpha=0.6, markersize=3, label="Noisy")
        plt.plot(x, X_time_smooth[:, k], '-', lw=2, label="Time-smoothed")
        plt.plot(x, X_full_smooth[:, k], '--', lw=2, label="Time + spatial smoothed")
        plt.xlabel("Spatial index")
        plt.ylabel("Field value")
        plt.title(f"Early-time snapshot k={k} (temporal smoothing boundary check)")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path / f"early_snapshot_k{k:03d}.png", dpi=150)
        plt.close()
    
    print(f"[OK] Saved {min(n_snapshots, X_noisy.shape[1])} early-time diagnostic snapshots -> {save_dir}/")


def fit_dmd(X_train, svd_rank):
    dmd = DMD(svd_rank=svd_rank, exact=True, tlsq_rank=0)
    dmd.fit(X_train)
    Phi = dmd.modes
    evals = dmd.eigs
    x0 = X_train[:, [0]]
    b0, *_ = np.linalg.lstsq(Phi, x0, rcond=None)
    return dmd, Phi, evals, b0


def fit_bopdmd(X_train, svd_rank, dt_eff):
    n_snapshots = X_train.shape[1]
    t_train = np.arange(n_snapshots) * dt_eff

    bop = BOPDMD(
        svd_rank=svd_rank, 
        num_trials=0, 
        trial_size=0.6,
        eig_constraints={'stable'}
    )
    bop.fit(X_train, t_train)
    Phi = bop.modes
    omega = bop.eigs
    evals = np.exp(omega * dt_eff)
    x0 = X_train[:, [0]]
    b0, *_ = np.linalg.lstsq(Phi, x0, rcond=None)
    return bop, Phi, evals, b0


def dmd_diagnostics(
    X,
    dt=None,
    svd_rank=None,     # user choice: -1/None = full, 0 = auto (here proxy = 99.9%),
    tlsq_rank=0,       # kept for signature parity; not applied here (diagnostics only)
    exact=True,        # signature parity
    outdir="dmd_diag_plots"  # where to save PNGs
):
    """
    Diagnostics for DMD stability & conditioning.
    New:
      - Save energy-capture plot (PNG)
      - Save eigenvalue maps for FULL and REDUCED systems (PNGs)
      - Print kappa(Phi_exact)
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ===== 1) Snapshot SVDs =====
    U, s, Vh = np.linalg.svd(X, full_matrices=False)
    Xp = X[:, 1:]
    Xm = X[:, :-1]
    Um, sm, Vhm = np.linalg.svd(Xm, full_matrices=False)

    def cum_energy(sig):
        e2 = (sig**2)
        return e2.cumsum() / (e2.sum() + 1e-30)

    print("=== Snapshot SVD (X) ===")
    print("Top 10 singular values:", s[:10])
    print("Min/Max sigma:", s.min(), s.max())
    print("cond2(X):", (s.max() / s.min()))
    print("Energy @ ranks [1,2,5,10,20]:",
          [cum_energy(s)[min(k-1, len(s)-1)] for k in [1,2,5,10,20]])

    # ---- Save energy capture (elbow) plot ----
    e = cum_energy(s)
    r_idx = np.arange(1, len(e)+1)
    plt.figure()
    plt.plot(r_idx, e, marker='o', linewidth=1)
    plt.xlabel("Retained rank r")
    plt.ylabel("Cumulative energy capture")
    plt.title("SVD energy capture vs rank")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    energy_png = outdir / "svd_energy_capture.png"
    plt.savefig(energy_png, dpi=160)
    plt.close()
    print(f"Saved: {energy_png}")

    print("\n=== Left block SVD (Xm = X[:, :-1]) ===")
    print("Top 10 singular values:", sm[:10])
    print("Min/Max sigma:", sm.min(), sm.max())
    print("cond2(Xm):", (sm.max() / sm.min()))

    # ===== 2) Choose reduced rank r =====
    if svd_rank is None or svd_rank < 0:
        r = len(sm)                  # FULL
    elif svd_rank == 0:
        # "auto": pick smallest r with >= 99.9% energy of Xm
        e_m = cum_energy(sm)
        r = int(np.searchsorted(e_m, 0.999) + 1)
    else:
        r = min(int(svd_rank), len(sm))

    # ===== 3) Build reduced operator (rank r) and FULL operator =====
    # Common factors for ranks:
    def build_atilde_eigs(Xm, Xp, Um, sm, Vhm, r_use):
        Ur = Um[:, :r_use]
        Sr = np.diag(sm[:r_use])
        Vr = Vhm[:r_use, :]
        Atilde = Ur.conj().T @ Xp @ Vr.conj().T @ np.linalg.inv(Sr)
        evals, W = np.linalg.eig(Atilde)
        core = Xp @ Vr.conj().T @ np.linalg.inv(Sr)      # used by exact modes
        Phi_exact = core @ W
        return evals, Phi_exact, core

    # Reduced (chosen r)
    evals_red, Phi_red, core_red = build_atilde_eigs(Xm, Xp, Um, sm, Vhm, r)

    # Full (no truncation)
    r_full = len(sm)
    evals_full, Phi_full, core_full = build_atilde_eigs(Xm, Xp, Um, sm, Vhm, r_full)

    # ===== 4) Condition metrics =====
    # core cond
    cs_red = np.linalg.svd(core_red, compute_uv=False)
    k_core_red = cs_red.max() / (cs_red.min() + 1e-30)

    ps_red = np.linalg.svd(Phi_red, compute_uv=False)
    kappa_Phi = ps_red.max() / (ps_red.min() + 1e-30)

    print(f"\n=== Reduced operator (rank r={r}) ===")
    print("cond2(S_r):", sm[:r].max() / (sm[:r].min() + 1e-30))
    print("cond2(core = Xp Vr^H Sr^{-1}):", k_core_red)
    print("kappa(Phi_exact):", kappa_Phi)

    # ===== 5) Eigenvalue plots (FULL vs REDUCED) =====
    def save_eig_scatter(evals, fname, title):
        plt.figure()
        plt.scatter(evals.real, evals.imag, s=18, alpha=0.8)
        # unit circle for reference
        th = np.linspace(0, 2*np.pi, 512)
        plt.plot(np.cos(th), np.sin(th), linewidth=1, alpha=0.5)
        plt.axhline(0, linewidth=0.5, alpha=0.5)
        plt.axvline(0, linewidth=0.5, alpha=0.5)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel("Re(λ)")
        plt.ylabel("Im(λ)")
        plt.title(title)
        plt.tight_layout()
        png = outdir / fname
        plt.savefig(png, dpi=160)
        plt.close()
        print(f"Saved: {png}")

    save_eig_scatter(evals_full, "eigs_full.png",    f"DMD eigenvalues (FULL, r={r_full})")
    save_eig_scatter(evals_red,  "eigs_reduced.png", f"DMD eigenvalues (REDUCED, r={r})")

    # ===== 6) Optional: continuous-time rates =====
    if dt is not None:
        omega_red = np.log(evals_red) / dt
        print("\n=== Reduced continuous-time rates (first 10) ===")
        print(omega_red[:10])

    # ===== 7) Reconstruction diagnostics (on reduced system) =====
    x0 = X[:, [0]]
    b0, *_ = np.linalg.lstsq(Phi_red, x0, rcond=None)
    m = X.shape[1]
    Vand = np.vander(evals_red, N=m, increasing=True)  # shape (r, m)
    Xhat = Phi_red @ (Vand * b0.flatten()[:, None])
    train_rel_err = np.linalg.norm(X - Xhat, ord='fro') / (np.linalg.norm(X, ord='fro') + 1e-30)
    print("\n=== Reconstruction (reduced) ===")
    print("Relative Frobenius training error:", train_rel_err)

    per_t = np.linalg.norm(X - Xhat, axis=0) / (np.linalg.norm(X, axis=0) + 1e-15)
    print("Worst per-snapshot relative error:", per_t.max(), "at t-index", per_t.argmax())

    # ===== 8) Vandermonde conditioning (reduced) =====
    vs = np.linalg.svd(Vand, compute_uv=False)
    print("\n=== Vandermonde over time grid (reduced) ===")
    print("cond2(Vand(evals_reduced, m)):", vs.max() / (vs.min() + 1e-30))

    return {
        "svd_s": s,
        "svd_sm": sm,
        "evals_full": evals_full,
        "evals_reduced": evals_red,
        "Phi_reduced": Phi_red,
        "kappa_Phi": kappa_Phi,
        "energy_curve": e,
        "train_rel_err": train_rel_err,
        "per_snapshot_rel_error": per_t,
        "plots": {
            "energy_capture": str(energy_png),
            "eigs_full": str(outdir / "eigs_full.png"),
            "eigs_reduced": str(outdir / "eigs_reduced.png"),
        }
    }



def rollout_full_field_full_length(Phi, evals, b0, n_x, n_int, N, M, dt_eff):
    n_aug = n_int + 1
    k_all = np.arange(N + M + 1)
    disc_aug = np.zeros((n_aug, N + M + 1), dtype=complex)
    cont_aug = np.zeros_like(disc_aug)
    omega = np.log(evals) / dt_eff
    for k in k_all:
        disc_aug[:, [k]] = Phi @ (b0 * (evals**k).reshape(-1, 1))
        cont_aug[:, [k]] = Phi @ (b0 * np.exp(omega * (k * dt_eff)).reshape(-1, 1))
    disc_int = disc_aug[:n_int, :]
    cont_int = cont_aug[:n_int, :]
    X_hat_disc_full = np.zeros((n_x, N + M + 1), dtype=float)
    X_hat_cont_full = np.zeros((n_x, N + M + 1), dtype=float)
    X_hat_disc_full[1:-1, :] = disc_int.real
    X_hat_cont_full[1:-1, :] = cont_int.real
    return X_hat_disc_full, X_hat_cont_full


# ==========================================================
# Entrypoint
# ==========================================================
if __name__ == "__main__":
    main()