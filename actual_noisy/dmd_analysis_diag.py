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
from pydmd import HankelDMD, DMD
from pathlib import Path
# from scipy.ndimage import gaussian_filter1d
# from scipy.ndimage import uniform_filter1d
# from scipy.signal import savgol_filter


# ==========================================================
# Main
# ==========================================================
def main():
    # ---------- User config ----------
    data_file   = "density2_tavg.dat"        # rows = time, columns = space
    time_file   = "density2_tavg_times.dat"  # 1D list of timestamps
    dt          = None                       # Will be computed from time_file if None
    decimate_q  = 10
    N           = 200
    M           = 200

    rank_mode   = "manual"     # "manual" or "energy"
    r_manual    = 20
    energy_thr  = 0.999

    # Smoothing parameters (commented out - using raw noisy data)
    # temporal_sigma = 2.0  # Gaussian temporal smoothing (sigma in time steps)
    # spatial_window_size = 10  # Uniform filter window size for spatial smoothing
    # spatial_sigma = 5.0   # Gaussian spatial smoothing (sigma in spatial points)
    # spatial_window = 20   # Spatial Savitzky-Golay window
    # spatial_polyorder = 3  # Spatial Savitzky-Golay polynomial order

    # Hankel DMD parameters
    hankel_d    = 5            # Hankel delay embedding dimension

    save_npz    = "dmd_rollout_outputs_ANALYSIS.npz"
    # ---------------------------------

    # 1) Load data and time, then build snapshot matrix (n_x, n_time)
    X_full_noisy, dt = load_ccp_data(data_file, time_file, dt)
    # X_full_noisy: shape (n_x, n_time) after transpose from original (n_time, n_x)

    # 2) Smoothing disabled - use noisy data directly
    X_full_time_smooth = X_full_noisy.copy()
    # Alternative temporal smoothing (commented):
    # X_full_time_smooth = np.stack([
    #     gaussian_filter1d(X_full_noisy[i, :], sigma=temporal_sigma, mode='nearest')
    #     for i in range(X_full_noisy.shape[0])
    # ], axis=0)

    # 3) Spatial smoothing disabled - use time-smoothed data (which is just noisy data)
    X_full_smooth = X_full_time_smooth.copy()
    # Alternative spatial smoothing options (commented):
    # X_full_smooth = np.stack([gaussian_filter1d(X_full_time_smooth[:, j], sigma=spatial_sigma) for j in range(X_full_time_smooth.shape[1])], axis=1)
    # X_full_smooth = np.stack([uniform_filter1d(X_full_time_smooth[:, j], size=spatial_window_size) for j in range(X_full_time_smooth.shape[1])], axis=1)
    # X_full_smooth = np.stack([savgol_filter(X_full_time_smooth[:, j], window_length=spatial_window, polyorder=spatial_polyorder) for j in range(X_full_time_smooth.shape[1])], axis=1)

    # 4) Decimate (all versions: noisy and smoothed)
    X_full_noisy_dec, dt_eff = decimate_series(X_full_noisy, dt, decimate_q)
    X_full_time_smooth_dec, _ = decimate_series(X_full_time_smooth, dt, decimate_q)
    X_full, dt_eff = decimate_series(X_full_smooth, dt, decimate_q)
    n_x, n_time = X_full.shape
    validate_train_rollout(N, M, n_time)

    # 5) Interior extraction (Dirichlet) + affine augmentation
    # Use noisy data (no smoothing)
    X_aug, n_int = build_interior_affine_augmented(X_full)

    # 5) Rank selection
    X_train = X_aug[:, :N+1]
    svd_rank, energy_captured = choose_svd_rank(X_train, rank_mode, r_manual, energy_thr)

    # 6) Fit Hankel DMD
    dmd, Phi, evals, b0 = fit_hankel_dmd(X_train, svd_rank, hankel_d)

    # 7) Extract original space modes from embedded space
    # Hankel DMD modes are in delay-embedded space (d * n_aug dimensions)
    # We need to extract the first n_aug components to get back to original space
    n_aug = X_aug.shape[0]  # Original augmented dimension
    Phi_original = Phi[:n_aug, :]  # Extract first n_aug rows from embedded modes

    # 8) Rollouts (Hankel DMD specific)
    X_hat_disc_full, X_hat_cont_full = rollout_hankel_dmd(
        Phi_original, evals, b0, n_x, n_int, N, M, dt_eff
    )

    # Truncate to available truth for comparisons
    t_max   = min(N + M, n_time - 1)
    Y_truth = X_full[:, :t_max+1]  # Fully smoothed data (decimated)
    Y_noisy = X_full_noisy_dec[:, :t_max+1]  # Original noisy data (decimated)
    Y_time_smooth = X_full_time_smooth_dec[:, :t_max+1]  # Time-smoothed only (decimated)
    Yhat_d  = X_hat_disc_full[:, :t_max+1]
    Yhat_c  = X_hat_cont_full[:, :t_max+1]

    # 9) Relative/absolute error Frobenius norms (per-time)
    # Errors vs smoothed data
    fro_truth_t = np.linalg.norm(Y_truth, axis=0)
    err_disc_abs_smooth_t = np.linalg.norm(Yhat_d - Y_truth, axis=0)
    err_cont_abs_smooth_t = np.linalg.norm(Yhat_c - Y_truth, axis=0)

    err_disc_rel_smooth_t = np.divide(err_disc_abs_smooth_t, fro_truth_t,
                                       out=np.zeros_like(err_disc_abs_smooth_t), where=fro_truth_t > 0)
    err_cont_rel_smooth_t = np.divide(err_cont_abs_smooth_t, fro_truth_t,
                                       out=np.zeros_like(err_cont_abs_smooth_t), where=fro_truth_t > 0)

    fro0_smooth = fro_truth_t[0] if fro_truth_t.size and fro_truth_t[0] != 0 else 1.0
    err_disc_norm0_smooth_t = err_disc_abs_smooth_t / fro0_smooth
    err_cont_norm0_smooth_t = err_cont_abs_smooth_t / fro0_smooth

    # Errors vs noisy data
    fro_noisy_t = np.linalg.norm(Y_noisy, axis=0)
    err_disc_abs_noisy_t = np.linalg.norm(Yhat_d - Y_noisy, axis=0)
    err_cont_abs_noisy_t = np.linalg.norm(Yhat_c - Y_noisy, axis=0)

    err_disc_rel_noisy_t = np.divide(err_disc_abs_noisy_t, fro_noisy_t,
                                     out=np.zeros_like(err_disc_abs_noisy_t), where=fro_noisy_t > 0)
    err_cont_rel_noisy_t = np.divide(err_cont_abs_noisy_t, fro_noisy_t,
                                     out=np.zeros_like(err_cont_abs_noisy_t), where=fro_noisy_t > 0)

    fro0_noisy = fro_noisy_t[0] if fro_noisy_t.size and fro_noisy_t[0] != 0 else 1.0
    err_disc_norm0_noisy_t = err_disc_abs_noisy_t / fro0_noisy
    err_cont_norm0_noisy_t = err_cont_abs_noisy_t / fro0_noisy

    # 10) Save all required data
    np.savez_compressed(
        save_npz,
        X_full_noisy=X_full_noisy,
        X_full_time_smooth=X_full_time_smooth,
        X_full_smooth=X_full_smooth,
        X_full_decimated=X_full,
        dt=dt, dt_eff=dt_eff, decimate_q=decimate_q,
        N=N, M=M, rank_mode=rank_mode, svd_rank=svd_rank,
        energy_captured=energy_captured,
        modes=Phi, evals=evals, amplitudes=b0,
        singular_values=np.array(getattr(dmd, 'singular_values', [])),
        X_hat_disc_full=X_hat_disc_full,
        X_hat_cont_full=X_hat_cont_full,
        Y_truth=Y_truth, Y_noisy=Y_noisy,
        Yhat_disc=Yhat_d, Yhat_cont=Yhat_c,
        t_decimated=np.arange(t_max + 1) * dt_eff,
        t_full_prediction_window=np.arange(N + M + 1) * dt_eff,
        t_max=t_max,
        err_disc_abs_smooth_t=err_disc_abs_smooth_t,
        err_cont_abs_smooth_t=err_cont_abs_smooth_t,
        err_disc_rel_smooth_t=err_disc_rel_smooth_t,
        err_cont_rel_smooth_t=err_cont_rel_smooth_t,
        err_disc_norm0_smooth_t=err_disc_norm0_smooth_t,
        err_cont_norm0_smooth_t=err_cont_norm0_smooth_t,
        err_disc_abs_noisy_t=err_disc_abs_noisy_t,
        err_cont_abs_noisy_t=err_cont_abs_noisy_t,
        err_disc_rel_noisy_t=err_disc_rel_noisy_t,
        err_cont_rel_noisy_t=err_cont_rel_noisy_t,
        err_disc_norm0_noisy_t=err_disc_norm0_noisy_t,
        err_cont_norm0_noisy_t=err_cont_norm0_noisy_t,
    )

    print(f"[OK] Analysis complete. Data saved to: {save_npz}")


# ==========================================================
# Helper functions
# ==========================================================
def load_ccp_data(data_path: str, time_path: str, dt: float = None) -> tuple[np.ndarray, float]:
    """
    Load CCP Vlasov data:
      - data_file: rows = time, columns = space
      - time_file: 1D array of times

    Returns
    =======
    X : np.ndarray, shape (n_x, n_time)
        Snapshot matrix suitable for DMD (SPACE x TIME).
    dt : float
        Effective time step (computed from time array if not provided).
    """
    data_path = Path(data_path)
    time_path = Path(time_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not time_path.exists():
        raise FileNotFoundError(f"Time file not found: {time_path}")

    U = np.loadtxt(data_path.as_posix())  # (n_time, n_x)
    if U.ndim == 1:
        U = U[None, :]

    # Transpose so we get (n_x, n_time) for DMD
    X = U.T.astype(float)

    t = np.loadtxt(time_path.as_posix())
    t = np.ravel(t)
    if dt is None:
        if t.size < 2:
            dt = 1.0
        else:
            dt = float(t[1] - t[0])

    return X, dt


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
    """Take interior points only (remove first/last in space) and append an all-ones row for affine augmentation."""
    X_int = X_full[1:-1, :]
    ones = np.ones((1, X_int.shape[1]))
    return np.vstack([X_int, ones]), X_int.shape[0]


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


def fit_dmd(X_train, svd_rank):
    """Standard DMD fit (commented out - using Hankel DMD instead)."""
    # dmd = DMD(svd_rank=svd_rank, exact=True, tlsq_rank=0)
    # dmd.fit(X_train)
    # Phi = dmd.modes
    # evals = dmd.eigs
    # x0 = X_train[:, [0]]
    # b0, *_ = np.linalg.lstsq(Phi, x0, rcond=None)
    # return dmd, Phi, evals, b0
    pass


def fit_hankel_dmd(X_train, svd_rank, d):
    """
    Fit Hankel DMD model.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training data matrix (n_x, n_time)
    svd_rank : int
        SVD rank for truncation
    d : int
        Hankel delay embedding dimension
        
    Returns:
    --------
    dmd : HankelDMD instance
    Phi : np.ndarray
        DMD modes
    evals : np.ndarray
        DMD eigenvalues
    b0 : np.ndarray
        Initial amplitudes
    """
    dmd = HankelDMD(svd_rank=svd_rank, exact=True, d=d)
    dmd.fit(X_train)
    Phi = dmd.modes
    evals = dmd.eigs
    # Use HankelDMD's built-in amplitudes property (handles delay embedding correctly)
    b0 = dmd.amplitudes
    return dmd, Phi, evals, b0


def rollout_hankel_dmd(Phi, evals, b0, n_x, n_int, N, M, dt_eff):
    """
    Rollout function specifically for Hankel DMD.
    
    Parameters:
    -----------
    Phi : np.ndarray
        DMD modes (already extracted from embedded space, shape: n_aug x r)
    evals : np.ndarray
        DMD eigenvalues
    b0 : np.ndarray
        Mode amplitudes (from HankelDMD.amplitudes)
    n_x : int
        Full spatial dimension (including boundaries)
    n_int : int
        Interior spatial dimension (excluding boundaries)
    N : int
        Number of training steps
    M : int
        Number of forecast steps
    dt_eff : float
        Effective time step
        
    Returns:
    --------
    X_hat_disc_full : np.ndarray
        Discrete-time rollout (n_x, N+M+1)
    X_hat_cont_full : np.ndarray
        Continuous-time rollout (n_x, N+M+1)
    """
    n_aug = n_int + 1
    k_all = np.arange(N + M + 1)
    disc_aug = np.zeros((n_aug, N + M + 1), dtype=complex)
    cont_aug = np.zeros_like(disc_aug)
    omega = np.log(evals) / dt_eff
    
    # Ensure b0 is 1D array for proper broadcasting
    b0 = np.ravel(b0)
    
    for k in k_all:
        # Discrete-time: x_k = Phi @ (b0 * lambda^k)
        disc_aug[:, k] = (Phi @ (b0 * (evals**k))).ravel()
        # Continuous-time: x_k = Phi @ (b0 * exp(omega * t))
        cont_aug[:, k] = (Phi @ (b0 * np.exp(omega * (k * dt_eff)))).ravel()
    
    # Extract interior (remove affine row)
    disc_int = disc_aug[:n_int, :]
    cont_int = cont_aug[:n_int, :]
    
    # Reconstruct full field with zero Dirichlet boundaries
    X_hat_disc_full = np.zeros((n_x, N + M + 1), dtype=float)
    X_hat_cont_full = np.zeros((n_x, N + M + 1), dtype=float)
    X_hat_disc_full[1:-1, :] = disc_int.real
    X_hat_cont_full[1:-1, :] = cont_int.real
    
    return X_hat_disc_full, X_hat_cont_full


def rollout_full_field_full_length(Phi, evals, b0, n_x, n_int, N, M, dt_eff):
    """
    Original rollout function for standard DMD (kept for reference/compatibility).
    """
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

