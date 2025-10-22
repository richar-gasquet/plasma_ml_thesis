import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from dmd_utils import DMD

# ---------- Plot helpers ----------

def plot_error_curve(t, rel_err, title, outpath):
    plt.figure(figsize=(7.2, 4.0))
    plt.plot(t, rel_err, lw=2)
    plt.xlabel("time")
    plt.ylabel("relative L2 error")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=140)
    plt.close()

def plot_snapshot_compare(x, U_true, U_hat, time_index, dt, title, outpath):
    j = int(time_index)
    plt.figure(figsize=(7.2, 4.0))
    plt.plot(x, np.real(U_true[:, j]), lw=2, label="True")
    plt.plot(x, np.real(U_hat[:, j]),  lw=1.5, ls="--", label="DMD")
    plt.xlabel("x")
    plt.ylabel("u(x, t)")
    plt.title(f"{title}  |  t = {j*dt:.4f}")
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=140)
    plt.close()

def ensure_outdir(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)


# ---------- Core pipeline ----------

def run_one_dataset(name: str, U: np.ndarray, x: np.ndarray, dt: float,
                    K: int, H: int, rank, tlsq: int, outroot: Path):
    """
    Apply standard DMD to advection-diffusion equation data.
    
    The advection-diffusion equation is:
    ∂u/∂t + v * ∂u/∂x = D * ∂²u/∂x²
    
    where:
    - u(x,t) is the concentration field
    - v is the advection velocity (constant)
    - D is the diffusion coefficient (constant)
    
    This is a linear PDE, making it ideal for DMD analysis. The equation can be
    discretized as: u(t+1) = A * u(t) where A is a linear operator that captures
    both advection and diffusion effects.
    
    DMD Approach:
    1. Learn the linear operator A from training data: u(t+1) ≈ A * u(t)
    2. Decompose A into modes with associated growth/decay rates and frequencies
    3. Use the learned modes to forecast future states
    
    Parameters
    ----------
    name : str
        Dataset identifier (e.g., "U_fd_clean", "U_fft_noisy")
    U : np.ndarray, shape (N, T)
        Snapshot matrix of concentration field over time
    x : np.ndarray, shape (N,)
        Spatial grid points
    dt : float
        Time step between snapshots
    K : int
        Number of training snapshots
    H : int
        Number of forecast steps beyond training
    rank : int or "auto"
        DMD truncation rank
    tlsq : int
        TLSQ denoising parameter for noisy data
    outroot : Path
        Output directory for results
    """
    n_space, T = U.shape
    K = min(K, T)  # safe-guard
    if K < 2:
        raise ValueError(f"K must be >= 2 (got {K}).")

    # Step 1: Train DMD on the concentration field data
    # For advection-diffusion, we can use standard DMD since the equation is linear
    # and autonomous (no external forcing or source terms)
    if rank == "auto":
        dmd = DMD(r=None, energy_thresh=0.999, tlsq=tlsq)
    else:
        dmd = DMD(r=int(rank), energy_thresh=0.999, tlsq=tlsq)

    # Learn the linear operator A such that u(t+1) ≈ A * u(t)
    # This captures both advection (transport) and diffusion (spreading) effects
    res = dmd.fit(U, dt, n_train=K)

    # Step 2: Forecast future states using the learned linear dynamics
    # Start from the last training snapshot to ensure continuity
    x_last = U[:, K-1]  # Last training snapshot
    U_future_hat = dmd.forecast(H, x_init=x_last)

    # Step 3: Combine training reconstruction and forecast for full timeline
    # This creates a complete prediction from t=0 to t=(K+H-1)*dt
    U_hat_full = np.hstack([res.Uhat_train, U_future_hat])

    # Step 4: Evaluate prediction quality against available ground truth
    # Limit evaluation to available data (may be shorter than full prediction)
    T_eval = min(T, K + H)  # Evaluation window
    U_true_eval = U[:, :T_eval]      # Ground truth
    U_pred_eval = U_hat_full[:, :T_eval]  # Prediction

    # Compute relative L2 error at each time step
    # This measures how well DMD captures the advection-diffusion dynamics
    rel_err = DMD.rel_l2_over_time(U_pred_eval, U_true_eval)
    t_eval = dt * np.arange(T_eval)  # Time points for evaluation

    # Save outputs
    ds_out = outroot / name
    ensure_outdir(ds_out)

    # Error curve
    plot_error_curve(t_eval, rel_err,
                     title=f"{name}: DMD (rank={'auto' if rank=='auto' else int(rank)}) tlsq={tlsq} | K={K}, H={H}",
                     outpath=ds_out / "error_curve.png")

    # A couple of snapshot comparisons
    mid = min(T_eval-1, max(1, K // 2))
    end = T_eval - 1
    plot_snapshot_compare(x, U_true_eval, U_pred_eval, mid, dt,
                          title=f"{name} (mid)",
                          outpath=ds_out / "snapshot_mid.png")
    plot_snapshot_compare(x, U_true_eval, U_pred_eval, end, dt,
                          title=f"{name} (end)",
                          outpath=ds_out / "snapshot_end.png")

    # Print a small text summary
    with open(ds_out / "summary.txt", "w") as f:
        f.write(f"Dataset: {name}\n")
        f.write(f"Shape U: {U.shape}\n")
        f.write(f"dt: {dt}\n")
        f.write(f"K (train): {K}, H (forecast): {H}\n")
        f.write(f"Rank: {rank}\n")
        f.write(f"TLSQ: {tlsq}\n")
        f.write(f"Train reconstruction rel L2 (last train col): "
                f"{DMD.rel_l2(res.Uhat_train[:, -1], U[:, K-1]):.6e}\n")
        f.write(f"Eval window rel L2 (mean): {float(np.mean(rel_err)):.6e}\n")
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
                        help="TLSQ parameter for noisy data. 0 = off; try 2, 5, or 10 for noise robustness.")
    parser.add_argument("--root", type=str, default=".", help="Directory containing x.txt, dt.txt, and U_*.txt files")
    parser.add_argument("--out", type=str, default="dmd_outputs", help="Directory to write outputs")
    args = parser.parse_args()

    root = Path(args.root)
    outroot = Path(args.out)
    ensure_outdir(outroot)

    expected = ["advection_diffusion/x.txt", "advection_diffusion/dt.txt", "advection_diffusion/U_fd.txt", "advection_diffusion/U_fft.txt", "advection_diffusion/U_fd_noisy.txt", "advection_diffusion/U_fft_noisy.txt"]
    missing = [p for p in expected if not (root / p).exists()]
    if missing:
        print("\n[!] Missing required data files in", root.resolve())
        for m in missing:
            print("   -", m)
        print("\nPlace your files in this directory and re-run. "
              "The code expects each U_*.txt to be shaped (N, T), i.e., columns are time snapshots.")
        return

    # Load
    x = np.loadtxt(root / "advection_diffusion/x.txt")
    dt = float(np.loadtxt(root / "advection_diffusion/dt.txt"))
    U_fd = np.loadtxt(root / "advection_diffusion/U_fd.txt")
    U_fft = np.loadtxt(root / "advection_diffusion/U_fft.txt")
    U_fd_noisy = np.loadtxt(root / "advection_diffusion/U_fd_noisy.txt")
    U_fft_noisy = np.loadtxt(root / "advection_diffusion/U_fft_noisy.txt")

    # Sanity: ensure shape (N, T). If they came as (T, N), transpose once.
    def ensure_N_by_T(U):
        if U.shape[0] == x.shape[0]:
            return U
        elif U.shape[1] == x.shape[0]:
            return U.T
        else:
            raise ValueError(f"Data shape {U.shape} doesn't match x length {x.shape[0]}.")

    U_fd = ensure_N_by_T(U_fd)
    U_fft = ensure_N_by_T(U_fft)
    U_fd_noisy = ensure_N_by_T(U_fd_noisy)
    U_fft_noisy = ensure_N_by_T(U_fft_noisy)

    # Run on clean datasets with vanilla DMD
    print("\n=== Clean datasets: standard DMD ===")
    run_one_dataset("U_fft_clean", U_fft, x, dt, args.K, args.H, rank=args.rank, tlsq=0, outroot=outroot)
    run_one_dataset("U_fd_clean",  U_fd,  x, dt, args.K, args.H, rank=args.rank, tlsq=0, outroot=outroot)

    # Run on noisy datasets with TLSQ-DMD
    print("\n=== Noisy datasets: TLSQ-DMD (tune --tlsq) ===")
    run_one_dataset("U_fft_noisy", U_fft_noisy, x, dt, args.K, args.H, rank=args.rank, tlsq=args.tlsq, outroot=outroot)
    run_one_dataset("U_fd_noisy",  U_fd_noisy,  x, dt, args.K, args.H, rank=args.rank, tlsq=args.tlsq, outroot=outroot)

    print(f"\nDone. Outputs in: {outroot.resolve()}")
    print(" - error_curve.png per dataset")
    print(" - snapshot_mid.png / snapshot_end.png per dataset")
    print(" - summary.txt per dataset")


if __name__ == "__main__":
    main()
