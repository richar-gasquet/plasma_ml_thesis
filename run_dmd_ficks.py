from pathlib import Path

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from dmd_utils import DMD


# ---------- Utilities ----------

def ensure_outdir(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)


def ensure_N_by_T(U, n_expected):
    """
    Ensure data is shaped (N, T). If it comes as (T, N) and N matches, transpose.
    """
    U = np.asarray(U)
    if U.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {U.shape}.")
    if U.shape[0] == n_expected:
        return U
    if U.shape[1] == n_expected:
        return U.T
    raise ValueError(f"Data shape {U.shape} doesn't match expected N={n_expected}.")


def broadcast_S(S, N, T):
    """
    Accept S with shape (N,) or (N, T). Broadcast as needed to (N, T).
    """
    S = np.asarray(S)
    if S.ndim == 1:
        if S.shape[0] != N:
            raise ValueError(f"S has shape {S.shape}, expected ({N},) for 1D input.")
        return np.tile(S[:, None], (1, T))
    elif S.ndim == 2:
        if S.shape == (N, T):
            return S
        if S.shape == (N, 1):
            return np.tile(S, (1, T))
        # Do not guess if the second dim equals T but the first dim is not N
        raise ValueError(f"S has shape {S.shape}, expected ({N}, T) or ({N},).")
    else:
        raise ValueError(f"S must be 1D or 2D, got shape {S.shape}.")


def plot_error_curve(t, rel_err, title, outpath):
    plt.figure(figsize=(7.2, 4.0))
    plt.plot(t, rel_err, lw=2)
    plt.xlabel("time")
    plt.ylabel("relative L2 error")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_snapshot_compare(x, U_true, U_hat, time_index, dt, title, outpath):
    """
    Overlay a true and predicted snapshot at a single time index.
    """
    plt.figure(figsize=(7.2, 4.0))
    plt.plot(x, U_true[:, time_index], lw=2, label="true")
    plt.plot(x, U_hat[:, time_index], lw=2, ls="--", label="pred")
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    t = time_index * dt
    plt.title(f"{title}  (t={t:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


# ---------- Core pipeline ----------

def run_ficks_dataset(name: str,
                      U: np.ndarray,
                      S: np.ndarray,
                      x: np.ndarray,
                      dt: float,
                      K: int,
                      H: int,
                      rank,
                      tlsq: int,
                      outroot: Path):
    """
    Apply DMD to Fick's second law (diffusion with source terms) using augmented state approach.
    
    Fick's second law describes diffusion with a source term:
    ∂u/∂t = D * ∇²u + S(x,t)
    
    where:
    - u(x,t) is the concentration field
    - D is the diffusion coefficient  
    - S(x,t) is the source term
    - The equation is linear in u but has a time-dependent forcing S(x,t)
    
    DMD Challenge:
    Standard DMD assumes u(t+1) = A*u(t), but Fick's law has:
    u(t+1) = A*u(t) + S(t+1)
    
    Augmented State Solution:
    We create an augmented state vector [u; S; 1] where:
    - u is the concentration field (N spatial points)
    - S is the source term (N spatial points) 
    - 1 is a constant bias term (1 point)
    
    This allows DMD to learn: [u(t+1); S(t+1); 1] = A_aug * [u(t); S(t); 1]
    
    During forecasting:
    1. Use DMD to predict the full augmented state
    2. Extract the u-component as the concentration prediction
    3. Inject the known S(t) and bias 1 at each time step
    4. This accounts for the time-dependent source while using linear DMD
    
    Parameters
    ----------
    name : str
        Dataset identifier for output files
    U : np.ndarray, shape (N, T)
        Concentration snapshots u(x,t) over time
    S : np.ndarray, shape (N,) or (N, T)  
        Source term S(x,t). If 1D, broadcast to all time steps
    x : np.ndarray, shape (N,)
        Spatial grid points
    dt : float
        Time step between snapshots
    K : int
        Number of training snapshots (first K columns of U)
    H : int
        Number of forecast steps beyond training
    rank : int or "auto"
        DMD truncation rank. "auto" uses energy-based selection
    tlsq : int
        TLSQ denoising parameter (0 = no denoising)
    outroot : Path
        Output directory for results
    """
    N, T = U.shape
    K = min(K, T)
    if K < 2:
        raise ValueError(f"K must be >= 2 (got {K}).")

    # Step 1: Prepare source term data
    # Broadcast S to (N, T) if it's 1D (constant in time) or keep as (N, T)
    # This ensures S has the same temporal dimension as U
    S_bt = broadcast_S(S, N, T)

    # Step 2: Create augmented state matrix [u; S; 1]
    # This is the key innovation for handling source terms in DMD
    # Shape: (2N + 1, T) where:
    # - First N rows: concentration field u(x,t)
    # - Next N rows: source term S(x,t) 
    # - Last row: constant bias term 1
    ones_row = np.ones((1, T))
    U_stacked = np.vstack([U, S_bt, ones_row])
    M = U_stacked.shape[0]  # Total augmented state dimension: 2N + 1

    # Step 3: Train DMD on the augmented state
    # The DMD will learn the linear operator A_aug such that:
    # [u(t+1); S(t+1); 1] ≈ A_aug * [u(t); S(t); 1]
    # This captures both the diffusion dynamics and the source coupling
    if rank == "auto":
        dmd = DMD(r=None, energy_thresh=0.999, tlsq=tlsq)
    else:
        dmd = DMD(r=int(rank), energy_thresh=0.999, tlsq=tlsq)

    res = dmd.fit(U_stacked, dt, n_train=K)

    # Step 4: Forecast H steps using the learned DMD model
    # The key challenge is that we need to inject the known source term S(t) 
    # at each time step, since DMD can't predict the external forcing
    
    # Initialize from the last training snapshot
    # Ensure the S and bias components are exactly correct at the transition point
    x_prev = U_stacked[:, K-1].copy()  # Last training augmented state
    x_prev[N:2*N] = S_bt[:, K-1]       # Inject exact S at time K-1
    x_prev[-1] = 1.0                    # Inject exact bias term

    # Storage for forecast results
    preds = np.zeros((M, H))  # Shape: (2N+1, H)
    
    # Iterative forecasting with source injection
    for j in range(H):
        # Use DMD to predict one step ahead from current augmented state
        step = dmd.forecast(1, x_init=x_prev)
        
        # Handle both 1D and 2D return formats from forecast
        if step.ndim == 1:
            xj = step
        else:
            xj = step[:, -1]
        
        # CRITICAL: Inject the known source term and bias at this time step
        # This is where we account for the external forcing S(x,t)
        # The DMD prediction for S and bias is discarded and replaced with truth
        idx = min(K + j, T - 1)  # Time index for source (clamp to available data)
        xj[N:2*N] = S_bt[:, idx]  # Inject exact S(x, t) at time K+j
        xj[-1] = 1.0              # Inject exact bias term
        
        # Store this prediction and use as initial condition for next step
        preds[:, j] = xj
        x_prev = xj

    # Step 5: Extract concentration predictions and evaluate performance
    # We only care about the u-component (concentration field) for evaluation
    # The S and bias components were just used to help DMD learn the dynamics
    
    # Extract u-component from training reconstruction
    Uhat_train_u = res.Uhat_train[:N, :]   # First N rows = concentration field
    
    # Extract u-component from forecast
    U_future_u = preds[:N, :]              # First N rows = predicted concentration
    
    # Combine training and forecast for full prediction timeline
    U_hat_full_u = np.hstack([Uhat_train_u, U_future_u])

    # Step 6: Evaluate prediction quality against ground truth
    # Limit evaluation to available truth data (may be shorter than prediction)
    T_eval = min(T, K + H)  # Evaluation window: training + available forecast
    U_true_eval = U[:, :T_eval]      # Ground truth concentration
    U_pred_eval = U_hat_full_u[:, :T_eval]  # Predicted concentration

    # Compute relative L2 error at each time step
    # This measures how well DMD captures the concentration evolution
    rel_err = DMD.rel_l2_over_time(U_pred_eval, U_true_eval)
    t_eval = dt * np.arange(T_eval)  # Time points for evaluation

    # Output directory for this dataset
    ds_out = outroot / name
    ensure_outdir(ds_out)

    # Plots
    plot_error_curve(t_eval, rel_err,
                     title=f"{name}: DMD on [u; S; 1] (rank={'auto' if rank=='auto' else int(rank)}) tlsq={tlsq} | K={K}, H={H}",
                     outpath=ds_out / "error_curve.png")
    # Snapshots: mid and end within the eval horizon
    mid = min(T_eval-1, max(1, K // 2))
    end = T_eval - 1
    plot_snapshot_compare(x, U_true_eval, U_pred_eval, mid, dt,
                          title=f"{name} (mid)",
                          outpath=ds_out / "snapshot_mid.png")
    plot_snapshot_compare(x, U_true_eval, U_pred_eval, end, dt,
                          title=f"{name} (end)",
                          outpath=ds_out / "snapshot_end.png")

    # Summary
    with open(ds_out / "summary.txt", "w") as f:
        f.write(f"name: {name}\n")
        f.write(f"K={K}, H={H}, dt={dt}\n")
        f.write(f"rank={'auto' if rank=='auto' else int(rank)}, tlsq={tlsq}\n")
        f.write(f"rank_used={res.r}\n")
        f.write("singular_values=" + np.array2string(res.singular_values, precision=4, separator=", ") + "\n")
        f.write(f"T_eval={T_eval}\n")
        f.write(f"Eval window rel L2 (mean):   {float(np.mean(rel_err)):.6e}\n")
        f.write(f"Eval window rel L2 (median): {float(np.median(rel_err)):.6e}\n")
        f.write(f"Eval window rel L2 (at end): {float(rel_err[-1]):.6e}\n")

    return {
        "rel_err": rel_err,
        "t_eval": t_eval,
        "U_pred_eval": U_pred_eval,
        "U_true_eval": U_true_eval,
        "rank_used": res.r,
        "singular_values": res.singular_values
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=1000, help="Number of initial steps to train on")
    parser.add_argument("--H", type=int, default=5000, help="Number of steps to forecast into the future")
    parser.add_argument("--rank", type=str, default="auto",
                        help="'auto' for energy-based truncation, or an integer rank (e.g. 20)")
    parser.add_argument("--tlsq", type=int, default=0,
                        help="TLSQ parameter. 0 = off; try 2, 5, or 10 for noise robustness.")
    parser.add_argument("--root", type=str, default="./diffusion_bounded_linear",
                        help="Directory containing x.txt, dt.txt, U.txt, and S.txt (or override with --u_file/--s_file)")
    parser.add_argument("--u_file", type=str, default="U.txt",
                        help="Filename for u snapshots inside --root (shape (N,T) or (T,N))")
    parser.add_argument("--s_file", type=str, default="U_source.txt",
                        help="Filename for source S inside --root (shape (N,) or (N,T))")
    parser.add_argument("--out", type=str, default="dmd_ficks_outputs", help="Directory to write outputs")
    parser.add_argument("--name", type=str, default="ficks",
                        help="Dataset name to tag outputs (used as subfolder)")
    args = parser.parse_args()

    root = Path(args.root)
    outroot = Path(args.out)
    ensure_outdir(outroot)

    # Load grid and time step
    x_path = root / "x.txt"
    dt_path = root / "dt.txt"
    # print(x_path)
    # print(dt_path)
    if not x_path.exists() or not dt_path.exists():
        raise FileNotFoundError(f"Expected x.txt and dt.txt in {root}.")

    x = np.loadtxt(x_path)
    dt = float(np.loadtxt(dt_path))

    # Load U and S
    U = np.loadtxt(root / args.u_file)
    # print(root / args.u_file)
    # If U was saved as (T, N), transpose to (N, T) using x length
    U = ensure_N_by_T(U, n_expected=x.shape[0])

    S = np.loadtxt(root / args.s_file)
    # S can be (N,) or (N, T). Broadcast later using N and T from U.
    N, T = U.shape

    # Run the pipeline
    print("\n=== Fick's: DMD on [u; S; 1] ===")
    run_ficks_dataset(args.name, U, S, x, dt, args.K, args.H, rank=args.rank, tlsq=args.tlsq, outroot=outroot)

    print(f"\nDone. Outputs in: {outroot.resolve()}")
    print(" - error_curve.png")
    print(" - snapshot_mid.png / snapshot_end.png")
    print(" - summary.txt")


if __name__ == "__main__":
    main()
'''

out_path = Path("/mnt/data/run_dmd_ficks.py")
out_path.write_text(code)
print(f"Saved: {out_path} ({out_path.stat().st_size} bytes)")
'''

