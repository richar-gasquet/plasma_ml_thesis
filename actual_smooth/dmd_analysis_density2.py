#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 1 (analysis only): PyDMD-based DMD analysis for CCP Vlasov data.

Dataset:
  - Data file  : "density2.dat"        (rows = time, columns = space)
  - Time file  : "density2_times.dat"  (list of timestamps)

What this script does (mirrors diffusion_bounded_linear/dmd_analysis_diag.py as closely as possible):
  1) Loads the data and **transposes** it so snapshots matrix has shape (n_x, n_time).
  2) Optionally temporally decimates the data.
  3) Extracts interior points (Dirichlet-style) and applies an affine augmentation.
  4) Selects the SVD rank (manual or energy-based).
  5) Fits a standard DMD model.
  6) Builds discrete- and continuous-time rollouts over a train+forecast window.
  7) Computes Frobenius error norms over time.
  8) Saves all needed arrays to a compressed NPZ file for plotting.
"""

import os
from pathlib import Path

import numpy as np
from pydmd import DMD


# ==========================================================
# Main
# ==========================================================
def main():
    # ---------- User config ----------
    data_file   = "density2.dat"        # rows = time, columns = space
    time_file   = "density2_times.dat"  # 1D list of timestamps

    # DMD / snapshot settings
    decimate_q  = 1        # temporal decimation factor (1 = no decimation)
    N           = 300      # number of train steps  (on decimated grid)
    M           = 600      # number of forecast steps (on decimated grid)

    rank_mode   = "manual"   # "manual" or "energy"
    r_manual    = 20
    energy_thr  = 0.999

    save_npz    = "dmd_density2_ANALYSIS.npz"
    # ---------------------------------

    # 1) Load data and time, then build snapshot matrix (n_x, n_time)
    X_full_noisy, dt = load_ccp_data(data_file, time_file)
    # X_full_noisy: shape (n_x, n_time) after transpose from original (n_time, n_x)

    # 2) Temporal decimation
    X_full, dt_eff = decimate_series(X_full_noisy, dt, decimate_q)
    n_x, n_time = X_full.shape
    validate_train_rollout(N, M, n_time)

    # 3) Interior extraction (Dirichlet-style) + affine augmentation
    X_aug, n_int = build_interior_affine_augmented(X_full)

    # 4) Rank selection
    X_train = X_aug[:, :N + 1]
    svd_rank, energy_captured = choose_svd_rank(X_train, rank_mode, r_manual, energy_thr)

    # 5) Fit DMD (discrete-time)
    dmd, Phi, evals, b0 = fit_dmd(X_train, svd_rank)

    # 6) Rollouts (discrete & continuous)
    X_hat_disc_full, X_hat_cont_full = rollout_full_field_full_length(
        Phi, evals, b0, n_x, n_int, N, M, dt_eff
    )

    # 7) Truncate to available truth for comparisons
    t_max   = min(N + M, n_time - 1)
    Y_truth = X_full[:, :t_max + 1]
    Yhat_d  = X_hat_disc_full[:, :t_max + 1]
    Yhat_c  = X_hat_cont_full[:, :t_max + 1]

    # 8) Frobenius error norms (per time)
    fro_truth_t = np.linalg.norm(Y_truth, axis=0)               # ||Y(t)||_F
    err_disc_abs_t = np.linalg.norm(Yhat_d - Y_truth, axis=0)   # ||E_disc(t)||_F
    err_cont_abs_t = np.linalg.norm(Yhat_c - Y_truth, axis=0)   # ||E_cont(t)||_F

    # Relative (divide by ||Y(t)||_F); safe divide
    err_disc_rel_t = np.divide(
        err_disc_abs_t,
        fro_truth_t,
        out=np.zeros_like(err_disc_abs_t),
        where=fro_truth_t > 0,
    )
    err_cont_rel_t = np.divide(
        err_cont_abs_t,
        fro_truth_t,
        out=np.zeros_like(err_cont_abs_t),
        where=fro_truth_t > 0,
    )

    # Normalized-to-initial (divide by ||Y(0)||_F)
    fro0 = fro_truth_t[0] if fro_truth_t.size and fro_truth_t[0] != 0 else 1.0
    err_disc_norm0_t = err_disc_abs_t / fro0
    err_cont_norm0_t = err_cont_abs_t / fro0

    # 9) Save all required data
    np.savez_compressed(
        save_npz,
        # Raw inputs & configuration
        X_full_original=X_full_noisy,     # (n_x, n_time) transposed from original
        X_full_decimated=X_full,          # (n_x, n_time_decimated)
        dt=dt,
        dt_eff=dt_eff,
        decimate_q=decimate_q,
        N=N,
        M=M,
        rank_mode=rank_mode,
        svd_rank=svd_rank,
        energy_captured=energy_captured,
        # DMD factors
        modes=Phi,
        evals=evals,
        amplitudes=b0,
        singular_values=np.array(getattr(dmd, "singular_values", [])),
        # Predictions (full field, full-length rollout 0..N+M)
        X_hat_disc_full=X_hat_disc_full,
        X_hat_cont_full=X_hat_cont_full,
        # Truth & truncated predictions for aligned comparisons (0..t_max)
        Y_truth=Y_truth,
        Yhat_disc=Yhat_d,
        Yhat_cont=Yhat_c,
        # Time axes
        t_decimated=np.arange(t_max + 1) * dt_eff,
        t_full_prediction_window=np.arange(N + M + 1) * dt_eff,
        t_max=t_max,
        # Error Frobenius norms (time-resolved)
        err_disc_abs_t=err_disc_abs_t,
        err_cont_abs_t=err_cont_abs_t,
        err_disc_rel_t=err_disc_rel_t,
        err_cont_rel_t=err_cont_rel_t,
        err_disc_norm0_t=err_disc_norm0_t,
        err_cont_norm0_t=err_cont_norm0_t,
    )

    print(f"[OK] CCP density2 DMD analysis complete. Data saved to: {save_npz}")


# ==========================================================
# Helper functions
# ==========================================================
def load_ccp_data(data_path: str, time_path: str) -> tuple[np.ndarray, float]:
    """
    Load CCP Vlasov data:
      - data_file: rows = time, columns = space
      - time_file: 1D array of times

    Returns
    =======
    X : np.ndarray, shape (n_x, n_time)
        Snapshot matrix suitable for DMD (SPACE x TIME).
    dt : float
        Effective time step (assumed uniform from time array).
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
    if t.size < 2:
        dt = 1.0
    else:
        dt = float(t[1] - t[0])

    return X, dt


def decimate_series(X: np.ndarray, dt: float, q: int) -> tuple[np.ndarray, float]:
    if q < 1 or int(q) != q:
        raise ValueError("decimate_q must be a positive integer.")
    q = int(q)
    return X[:, ::q], dt * q


def validate_train_rollout(N: int, M: int, n_time: int) -> None:
    if not (1 <= N < n_time):
        raise ValueError(f"N must be in [1, n_time-1]. Got N={N}, n_time={n_time}.")
    if M < 1:
        raise ValueError("M must be >= 1.")
    if N + M >= n_time:
        print(f"[WARN] Ground truth truncates rollout: N+M={N+M} >= n_time={n_time}.")


def build_interior_affine_augmented(X_full: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Take interior points only (remove first/last in space) and append an all-ones row
    for affine (bias) augmentation, mirroring the diffusion_bounded_linear script.
    """
    X_int = X_full[1:-1, :]
    ones = np.ones((1, X_int.shape[1]))
    return np.vstack([X_int, ones]), X_int.shape[0]


def choose_svd_rank(
    X_train: np.ndarray, mode: str, r_manual: int, energy_thr: float
) -> tuple[int, float]:
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


def cumulative_energy(svals: np.ndarray, r: int) -> float:
    r = max(0, min(r, svals.size))
    if r == 0:
        return 0.0
    e2 = svals**2
    den = e2.sum()
    return 0.0 if den == 0.0 else float(e2[:r].sum() / den)


def fit_dmd(X_train: np.ndarray, svd_rank: int):
    """
    Standard DMD fit (discrete-time), mirroring the helper in dmd_analysis_diag.py.
    """
    dmd = DMD(svd_rank=svd_rank, exact=True, tlsq_rank=0)
    dmd.fit(X_train)
    Phi = dmd.modes
    evals = dmd.eigs
    x0 = X_train[:, [0]]
    b0, *_ = np.linalg.lstsq(Phi, x0, rcond=None)
    return dmd, Phi, evals, b0


def rollout_full_field_full_length(
    Phi: np.ndarray,
    evals: np.ndarray,
    b0: np.ndarray,
    n_x: int,
    n_int: int,
    N: int,
    M: int,
    dt_eff: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build discrete- and continuous-time rollouts on the **full** field (including
    Dirichlet endpoints) over times k = 0..N+M, mirroring the diffusion script.
    """
    n_aug = n_int + 1  # interior + bias row
    k_all = np.arange(N + M + 1)

    disc_aug = np.zeros((n_aug, N + M + 1), dtype=complex)
    cont_aug = np.zeros_like(disc_aug)

    omega = np.log(evals) / dt_eff
    for k in k_all:
        disc_aug[:, [k]] = Phi @ (b0 * (evals**k).reshape(-1, 1))
        cont_aug[:, [k]] = Phi @ (b0 * np.exp(omega * (k * dt_eff)).reshape(-1, 1))

    # strip off affine row, re-pad zeros at boundaries
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


