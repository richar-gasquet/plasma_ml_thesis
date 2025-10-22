import numpy as np
import sys
import os
import pickle
from pathlib import Path
import shutil
import matplotlib.pyplot as plt


#Main loop
def main():

	x     = np.loadtxt("x.txt")
	U_fd  = np.loadtxt("U_fd.txt")
	U_fft = np.loadtxt("U_fft.txt")
	U_fd_noisy  = np.loadtxt("U_fd_noisy.txt")
	U_fft_noisy = np.loadtxt("U_fft_noisy.txt")	
	dt    = float(np.loadtxt("dt.txt"))

	#Saving frames
	paths = save_advecdiff_plots(x, dt, U_fd, U_fft, outdir="frames", nprint=10)
	print(f"Saved {len(paths)} frames to: {paths[0].parent}")

	#Saving frames noisy
	paths = save_advecdiff_plots(x, dt, U_fd_noisy, U_fft_noisy, outdir="frames_noisy", nprint=10)
	print(f"Saved {len(paths)} frames to: {paths[0].parent}")	





#Saving output
def save_advecdiff_plots(x, dt, U_fd, U_fft, outdir, dpi=120, nprint=1):
	"""
	Save PNGs comparing FD and FFT solutions, skipping according to nprint.

	Parameters
	----------
	x : (N,) array
		Spatial grid (periodic). Used as x-axis for plotting.
	dt : float
		Time step between successive columns in U_fd/U_fft.
	U_fd : (N, T) array
		Snapshot matrix from the finite-difference solver (columns are times).
	U_fft : (N, T) array
		Snapshot matrix from the FFT/analytical evolution (columns are times).
	outdir : str or Path
		Directory to write PNGs into. Created if it doesn't exist.
	prefix : str, optional
		Filename prefix for saved frames (default: "frame").
	dpi : int, optional
		Resolution for saved PNGs.
	nprint : int, optional
		Save every nprint-th snapshot (default: 1, i.e. save all).

	Returns
	-------
	paths : list of Path
		File paths of the saved figures.
	"""
	outdir = Path(outdir)
	if outdir.exists():
		shutil.rmtree(outdir)
	outdir.mkdir(parents=True, exist_ok=True)

	U_fd_real  = np.real(U_fd)
	U_fft_real = np.real(U_fft)

	y_min = np.min([U_fd_real.min(), U_fft_real.min()])
	y_max = np.max([U_fd_real.max(), U_fft_real.max()])
	y_pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
	y_min -= y_pad
	y_max += y_pad

	T = U_fd_real.shape[1]
	tvals = dt * np.arange(T)

	# indices we will actually print
	indices = list(range(0, T, nprint))
	nframes = len(indices)

	paths = []
	for count, j in enumerate(indices, start=1):
		print(f"[frame {count} of {nframes}] saving snapshot j={j}")

		fig, ax = plt.subplots(figsize=(6.5, 3.4))
		ax.plot(x, U_fft_real[:, j], lw=2, label="FFT (analytic)")
		ax.plot(x, U_fd_real[:, j],  lw=1.5, ls="--", label="FD (numerical)")

		ax.set_xlim(x[0], x[-1])
		ax.set_ylim(y_min, y_max)
		ax.set_xlabel("x")
		ax.set_ylabel("u(x, t)")
		ax.set_title(f"Advection–Diffusion | t = {tvals[j]:.4f}  (dt = {dt:.3e})")
		#ax.grid(True, alpha=0.3)
		ax.legend(loc="best", frameon=True)

		fname = outdir / f"{j:04d}.png"
		fig.tight_layout()
		fig.savefig(fname, dpi=dpi)
		plt.close(fig)
		paths.append(fname)

	return paths





### MAIN FUNCTION ###

###*****************MAIN******************###
if __name__ == "__main__":
    main()