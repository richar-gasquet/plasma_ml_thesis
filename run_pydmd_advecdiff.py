import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pydmd import DMD as PYDMD



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
    plt.plot(x, np.real(U_hat[:, j]),  lw=1.5, ls="--", label="PyDMD")
    plt.xlabel("x")
    plt.ylabel("u(x, t)")
    plt.title(f"{title}  |  t = {j*dt:.4f}")
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=140)
    plt.close()


def ensure_outdir(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)


# ---------- Utilities ----------

def rel_l2(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    num = np.linalg.norm(a - b)
    den = np.linalg.norm(b)
    return float(num / (den + eps))


def rel_l2_over_time(U_pred: np.ndarray, U_true: np.ndarray) -> np.ndarray:
    assert U_pred.shape == U_true.shape
    T = U_pred.shape[1]
    errs = np.empty(T)
    for j in range(T):
        errs[j] = rel_l2(U_pred[:, j], U_true[:, j])
    return errs


# ---------- Core pipeline (PyDMD) ----------

def run_one_dataset_pydmd(name: str, U: np.ndarray, x: np.ndarray, dt: float,
                          K: int, H: int, rank, tlsq: int, outroot: Path):
    """
    Apply PyDMD to advection-diffusion data, reconstruct and forecast (rollout),
    and save comparison/error plots.

    Parameters
    ----------
    name : str
        Dataset identifier (e.g., "U_fd_clean", "U_fft_noisy")
    U : np.ndarray, shape (N, T)
        Snapshot matrix (columns are time snapshots)
    x : np.ndarray, shape (N,)
        Spatial grid
    dt : float
        Time step
    K : int
        Number of training snapshots
    H : int
        Number of forecast steps beyond training
    rank : int or "auto"
        PyDMD svd_rank. If "auto", we use 0 (PyDMD auto truncation)
    tlsq : int
        TLSQ denoising parameter for PyDMD
    outroot : Path
        Output directory
    """
    n_space, T = U.shape
    K = min(K, T)
    if K < 2:
        raise ValueError(f"K must be >= 2 (got {K}).")

    # Configure PyDMD
    if rank == "auto":
        svd_rank = 0  # PyDMD convention: 0 means 'auto'
    else:
        svd_rank = int(rank)

    dmd = PYDMD(svd_rank=svd_rank, tlsq=int(tlsq), exact=True)

    # Fit on training window
    U_train = U[:, :K]
    dmd.fit(U_train)

    # Set time info to allow forecasting beyond training
    # PyDMD uses integer-based time indices by default; attach actual dt
    dmd.dmd_time['dt'] = dt
    dmd.dmd_time['t0'] = 0
    dmd.dmd_time['tend'] = K - 1  # training reconstruction

    # Reconstruction on training window
    Uhat_train = dmd.reconstructed_data

    # Forecast H steps into the future
    dmd.dmd_time['tend'] = (K - 1) + H
    Uhat_full = dmd.reconstructed_data

    # Ensure shapes (N, K) and (N, K+H)
    assert Uhat_train.shape[0] == n_space
    assert Uhat_full.shape[0] == n_space

    # Evaluate prediction against ground truth for the overlapping window
    T_eval = min(T, K + H)
    U_true_eval = U[:, :T_eval]
    U_pred_eval = Uhat_full[:, :T_eval]

    rel_err = rel_l2_over_time(U_pred_eval, U_true_eval)
    t_eval = dt * np.arange(T_eval)

    # Save outputs
    ds_out = outroot / name
    ensure_outdir(ds_out)

    title = f"{name}: PyDMD (rank={'auto' if rank=='auto' else int(rank)}) tlsq={tlsq} | K={K}, H={H}"
    plot_error_curve(t_eval, rel_err, title=title, outpath=ds_out / "error_curve.png")

    mid = min(T_eval-1, max(1, K // 2))
    end = T_eval - 1
    plot_snapshot_compare(x, U_true_eval, U_pred_eval, mid, dt,
                          title=f"{name} (mid)", outpath=ds_out / "snapshot_mid.png")
    plot_snapshot_compare(x, U_true_eval, U_pred_eval, end, dt,
                          title=f"{name} (end)", outpath=ds_out / "snapshot_end.png")

    with open(ds_out / "summary.txt", "w") as f:
        f.write(f"Dataset: {name}\n")
        f.write(f"Shape U: {U.shape}\n")
        f.write(f"dt: {dt}\n")
        f.write(f"K (train): {K}, H (forecast): {H}\n")
        f.write(f"Rank: {rank}\n")
        f.write(f"TLSQ: {tlsq}\n")
        f.write(f"Eval window rel L2 (mean): {float(np.mean(rel_err)):.6e}\n")
        f.write(f"Eval window rel L2 (median): {float(np.median(rel_err)):.6e}\n")
        f.write(f"Eval window rel L2 (at end): {float(rel_err[-1]):.6e}\n")

    return {
        "rel_err": rel_err,
        "t_eval": t_eval,
        "U_pred_eval": U_pred_eval,
        "U_true_eval": U_true_eval,
        "rank_used": svd_rank,
        "singular_values": getattr(dmd, 'singular_values', None),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=1000, help="Number of initial steps to train on")
    parser.add_argument("--H", type=int, default=5000, help="Number of steps to forecast into the future")
    parser.add_argument("--rank", type=str, default="auto",
                        help="'auto' for PyDMD auto truncation, or an integer rank (e.g. 20)")
    parser.add_argument("--tlsq", type=int, default=0,
                        help="TLSQ parameter for noisy data. 0 = off; try 2, 5, or 10 for noise robustness.")
    parser.add_argument("--root", type=str, default=".", help="Directory containing x.txt, dt.txt, and U_*.txt files")
    parser.add_argument("--out", type=str, default="dmd_outputs_pydmd", help="Directory to write outputs")
    args = parser.parse_args()

    root = Path(args.root)
    outroot = Path(args.out)
    ensure_outdir(outroot)

    expected = [
        "advection_diffusion/x.txt",
        "advection_diffusion/dt.txt",
        "advection_diffusion/U_fd.txt",
        "advection_diffusion/U_fft.txt",
        "advection_diffusion/U_fd_noisy.txt",
        "advection_diffusion/U_fft_noisy.txt",
    ]
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

    # Ensure (N, T)
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

    # Clean datasets with vanilla PyDMD
    print("\n=== Clean datasets: PyDMD ===")
    run_one_dataset_pydmd("U_fft_clean", U_fft, x, dt, args.K, args.H, rank=args.rank, tlsq=0, outroot=outroot)
    run_one_dataset_pydmd("U_fd_clean",  U_fd,  x, dt, args.K, args.H, rank=args.rank, tlsq=0, outroot=outroot)

    # Noisy datasets with TLSQ
    print("\n=== Noisy datasets: PyDMD with TLSQ (tune --tlsq) ===")
    run_one_dataset_pydmd("U_fft_noisy", U_fft_noisy, x, dt, args.K, args.H, rank=args.rank, tlsq=args.tlsq, outroot=outroot)
    run_one_dataset_pydmd("U_fd_noisy",  U_fd_noisy,  x, dt, args.K, args.H, rank=args.rank, tlsq=args.tlsq, outroot=outroot)

    print(f"\nDone. Outputs in: {outroot.resolve()}")
    print(" - error_curve.png per dataset")
    print(" - snapshot_mid.png / snapshot_end.png per dataset")
    print(" - summary.txt per dataset")


if __name__ == "__main__":
    main()


