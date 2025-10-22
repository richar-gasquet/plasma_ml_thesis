from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple


@dataclass
class DMDResult:
    Phi: np.ndarray         # DMD modes  (n_space, r)
    lambda_: np.ndarray     # Discrete-time eigenvalues (r,)
    omega: np.ndarray       # Continuous-time eigenvalues (r,)
    b: np.ndarray           # Mode amplitudes (r,)
    r: int                  # Truncation rank actually used
    Uhat_train: np.ndarray  # Reconstruction on training window (n_space, n_train)
    singular_values: np.ndarray  # Singular values of X (pre-truncation)


class DMD:
    """
    Dynamic Mode Decomposition (DMD) implementation with TLSQ denoising capability.
    
    DMD is a data-driven method that decomposes high-dimensional dynamical systems
    into low-dimensional modes with associated growth/decay rates and frequencies.
    The method finds the best linear operator A that maps X to Y in the least-squares sense:
    Y ≈ A*X, where X and Y are snapshot matrices.
    
    Mathematical Framework:
    - Given snapshots X = [x_0, x_1, ..., x_{m-1}] and Y = [x_1, x_2, ..., x_m]
    - DMD finds A such that Y = A*X + R (where R is residual)
    - The eigendecomposition of A gives: A*Phi = Phi*Lambda
    - Each mode Phi_i evolves as: Phi_i(t) = Phi_i(0) * exp(omega_i * t)
    - where omega_i = log(lambda_i) / dt are continuous-time eigenvalues
    
    TLSQ (Total Least Squares) denoising helps with noisy data by:
    - Forming augmented matrix Z = [X; Y] 
    - Discarding smallest singular directions that likely contain noise
    - Projecting X and Y onto the cleaner subspace
    
    Parameters
    ----------
    r : int or None
        Truncation rank. If None, we pick it by energy threshold `energy_thresh`.
        Lower rank = more compression but potential loss of dynamics.
    energy_thresh : float
        If r is None, keep the smallest r such that sum(sv[:r]^2)/sum(sv^2) >= energy_thresh.
        Typical values: 0.99 (99% energy), 0.999 (99.9% energy)
    tlsq : int
        TLSQ parameter; number of smallest right-singular vectors of [X; Y] to drop.
        Use 0 for vanilla DMD (no TLSQ). Typical choices: 0, 2, 5, 10.
        Higher values = more aggressive denoising but potential loss of signal.
    """

    def __init__(self, r: Optional[int] = None, energy_thresh: float = 0.999, tlsq: int = 0):
        self.r = r
        self.energy_thresh = energy_thresh
        self.tlsq = max(0, int(tlsq))
        self._fitted: bool = False
        self._res: Optional[DMDResult] = None
        self.dt: Optional[float] = None

    @staticmethod
    def _svd_truncate_by_energy(s: np.ndarray, energy_thresh: float) -> int:
        """Return the minimal rank r such that cumulative energy >= energy_thresh."""
        if s.size == 0:
            return 0
        energy = np.cumsum(s**2) / np.sum(s**2)
        r = int(np.searchsorted(energy, energy_thresh) + 1)
        r = max(r, 1)
        return r

    def _apply_tlsq(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply TLSQ denoising:
            - Form augmented matrix Z = [X; Y]
            - Compute SVD Z = U Σ V*
            - Discard the last `tlsq` columns of V (the smallest singular directions)
            - Project X and Y onto the retained subspace: Xp = X V1 V1*, Yp = Y V1 V1*
        This follows PyDMD's TLSQ-DMD idea in spirit and is very effective for noisy data.
        """
        if self.tlsq <= 0:
            return X, Y

        Z = np.vstack([X, Y])   # shape ((2n), m)
        U, s, Vh = np.linalg.svd(Z, full_matrices=False)
        V = Vh.conj().T
        m = V.shape[0]
        r_tls = max(1, m - self.tlsq)  # retain this many directions
        V1 = V[:, :r_tls]
        # Project
        P = V1 @ V1.conj().T
        Xp = X @ P
        Yp = Y @ P
        return Xp, Yp

    def fit(self, U: np.ndarray, dt: float, n_train: int) -> DMDResult:
        """
        Fit DMD model on the first `n_train` columns of the snapshot matrix.
        
        This method implements the core DMD algorithm:
        1. Extract training data: X = U[:, :n_train-1], Y = U[:, 1:n_train]
        2. Apply TLSQ denoising if requested (for noisy data)
        3. Compute SVD of X: X = U_svd * S * V^H
        4. Truncate to rank r (either specified or energy-based)
        5. Build low-dimensional operator A_tilde in the POD subspace
        6. Compute eigenvalues and eigenvectors of A_tilde
        7. Project back to full space to get DMD modes
        8. Compute continuous-time eigenvalues: omega = log(lambda) / dt
        9. Fit mode amplitudes to initial condition
        
        The resulting model can reconstruct the training data and forecast future states
        using the linear evolution: u(t) = sum_i b_i * Phi_i * exp(omega_i * t)

        Parameters
        ----------
        U : np.ndarray, shape (n_space, n_time)
            Snapshot matrix where columns are time snapshots and rows are spatial points.
            Each column U[:, j] represents the state at time t = j*dt
        dt : float
            Time step between consecutive snapshots. Used to convert discrete-time
            eigenvalues to continuous-time eigenvalues.
        n_train : int
            Number of initial time steps to use for training (must be >= 2).
            More training data generally leads to better mode identification.

        Returns
        -------
        DMDResult
            Contains all DMD components: modes (Phi), eigenvalues (lambda_, omega),
            amplitudes (b), and reconstruction quality metrics.
        """
        assert U.ndim == 2, "U must be 2D: (n_space, n_time)"
        assert n_train >= 2, "n_train must be >= 2"
        n_space, n_time = U.shape
        n_train = min(n_train, n_time)

        # Step 1: Extract training data matrices
        # X contains snapshots [0, 1, ..., n_train-2], Y contains [1, 2, ..., n_train-1]
        # This creates the time-shifted pairs needed for DMD: Y ≈ A*X
        X = U[:, :n_train-1]  # Shape: (n_space, n_train-1)
        Y = U[:, 1:n_train]   # Shape: (n_space, n_train-1)

        # Step 2: Apply TLSQ denoising for noisy data
        # This projects X and Y onto a cleaner subspace by discarding
        # the smallest singular directions of the augmented matrix [X; Y]
        Xd, Yd = self._apply_tlsq(X, Y)

        # Step 3: Compute SVD of the (possibly denoised) data matrix X
        # X = U_svd * S * V^H, where U_svd are spatial modes, S are singular values,
        # and V^H are temporal modes
        Ux, s, Vhx = np.linalg.svd(Xd, full_matrices=False)

        # Step 4: Determine truncation rank
        # Either use specified rank or choose based on energy threshold
        # Higher rank captures more dynamics but may include noise
        if self.r is None:
            r = self._svd_truncate_by_energy(s, self.energy_thresh)
        else:
            r = min(self.r, s.size)
            r = max(1, r)

        # Extract truncated SVD components
        U_r  = Ux[:, :r]      # Truncated spatial modes: (n_space, r)
        S_r  = np.diag(s[:r]) # Truncated singular values: (r, r)
        V_rh = Vhx[:r, :]     # Truncated temporal modes: (r, n_train-1)

        # Step 5: Build low-dimensional linear operator in POD subspace
        # Mathematical formula: A_tilde = U_r^H * Y * V_r * S_r^{-1}
        # In numpy terms: A_tilde = U_r.conj().T @ Yd @ V_rh.conj().T @ S_r^{-1}
        # Note: V_rh from SVD is actually V_r^H, so V_rh.conj().T gives us V_r
        # This is the best linear map in the r-dimensional POD subspace
        A_tilde = U_r.conj().T @ Yd @ V_rh.conj().T @ np.linalg.inv(S_r)

        # Step 6: Compute eigendecomposition of the low-dimensional operator
        # A_tilde * W = W * Lambda, where Lambda contains discrete-time eigenvalues
        lambda_, W = np.linalg.eig(A_tilde)  # Shape: (r,) and (r, r)

        # Step 7: Project DMD modes back to full space
        # Phi = Y * V_r^H * S_r^{-1} * W
        # Each column of Phi is a DMD mode in the full spatial domain
        Phi = Yd @ V_rh.conj().T @ np.linalg.inv(S_r) @ W  # Shape: (n_space, r)

        # Step 8: Convert discrete-time to continuous-time eigenvalues
        # omega = log(lambda) / dt gives growth/decay rates and frequencies
        # Real part: growth/decay rate, Imaginary part: frequency
        with np.errstate(divide='ignore', invalid='ignore'):
            omega = np.log(lambda_) / dt  # Shape: (r,)

        # Step 9: Fit mode amplitudes to initial condition
        # Solve Phi * b = x0 for b, where x0 is the first snapshot
        # This determines how much each mode contributes to the initial state
        x0 = U[:, 0]  # First snapshot
        b = np.linalg.lstsq(Phi, x0, rcond=None)[0]  # Shape: (r,)

        # Step 10: Reconstruct training data to verify model quality
        # This helps assess how well the DMD model captures the training dynamics
        t_train = dt * np.arange(n_train)
        Uhat_train = self._reconstruct(Phi, omega, b, t_train)

        self._fitted = True
        self._res = DMDResult(Phi=Phi, lambda_=lambda_, omega=omega, b=b,
                              r=r, Uhat_train=Uhat_train, singular_values=s)
        self.dt = dt
        return self._res

    @staticmethod
    def _reconstruct(Phi: np.ndarray, omega: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Reconstruct snapshots at times t using the DMD model.
        
        This implements the core DMD reconstruction formula:
        u(t) = sum_{i=1}^r b_i * Phi_i * exp(omega_i * t)
        
        where:
        - Phi_i is the i-th DMD mode (spatial pattern)
        - omega_i is the i-th continuous-time eigenvalue (complex)
        - b_i is the i-th mode amplitude
        - r is the number of modes
        
        The reconstruction assumes linear evolution in time. Each mode evolves
        independently with its own growth/decay rate and frequency.

        Parameters
        ----------
        Phi : np.ndarray, shape (n_space, r)
            DMD modes matrix. Each column is a spatial mode.
        omega : np.ndarray, shape (r,)
            Continuous-time eigenvalues. Complex values where:
            - Real part: growth/decay rate (positive = growing, negative = decaying)
            - Imaginary part: frequency (oscillation rate)
        b : np.ndarray, shape (r,)
            Mode amplitudes. Determines how much each mode contributes.
        t : np.ndarray, shape (n_time,)
            Time points at which to reconstruct the solution.

        Returns
        -------
        Uhat : np.ndarray, shape (n_space, n_time)
            Reconstructed snapshots. Each column is the state at the corresponding time.
        """
        n_modes = Phi.shape[1]
        
        # Compute time evolution for each mode: exp(omega_i * t) for all i, t
        # This creates a (r, n_time) matrix where entry (i,j) is exp(omega_i * t_j)
        time_dynamics = np.exp(np.outer(omega, t))  # shape (n_modes, len(t))
        
        # Weight each mode by its amplitude and sum over all modes
        # Phi @ (time_dynamics * b[:, None]) gives the reconstruction
        return Phi @ (time_dynamics * b[:, None])

    def forecast(self, n_future: int, x_init: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forecast n_future steps beyond the training window using the learned DMD model.
        
        The forecast uses the linear evolution equation:
        u(t) = sum_i b_i * Phi_i * exp(omega_i * t)
        
        where:
        - Phi_i are the DMD modes (spatial patterns)
        - omega_i are the continuous-time eigenvalues (growth/decay + frequency)
        - b_i are the mode amplitudes (determined from initial condition)
        
        The forecast assumes the system continues to evolve linearly according to
        the dynamics learned during training. This may not hold for:
        - Nonlinear systems beyond the training regime
        - Systems with changing parameters
        - Systems approaching different attractors

        Parameters
        ----------
        n_future : int
            Number of time steps to forecast into the future.
            Each step corresponds to dt time units.
        x_init : np.ndarray or None
            Initial condition for the forecast. If provided, mode amplitudes
            are re-fitted to this state. If None, uses the same amplitudes
            from training (fitted to the first training snapshot).

        Returns
        -------
        Uhat_future : np.ndarray, shape (n_space, n_future)
            Forecasted snapshots. Each column is a predicted state at time
            t = (training_end + i*dt) for i = 1, 2, ..., n_future
        """
        assert self._fitted and self._res is not None and self.dt is not None, "Call fit() first."
        Phi, omega, b = self._res.Phi, self._res.omega, self._res.b

        # Re-fit mode amplitudes if new initial condition provided
        if x_init is not None:
            b = np.linalg.lstsq(Phi, x_init, rcond=None)[0]

        # Time points for forecast: start from t=0 (not dt) for proper continuation
        # This ensures the forecast starts from the correct time reference
        t_future = self.dt * np.arange(1, n_future + 1)  # [dt, 2*dt, ..., n_future*dt]
        Uhat_future = self._reconstruct(Phi, omega, b, t_future)
        return Uhat_future

    @staticmethod
    def rel_l2(a: np.ndarray, b: np.ndarray, axis=None, eps: float = 1e-12) -> float:
        """Relative L2 error: ||a-b|| / ||b||."""
        num = np.linalg.norm(a - b, axis=axis)
        den = np.linalg.norm(b, axis=axis)
        return float(num / (den + eps))

    @staticmethod
    def rel_l2_over_time(U_pred: np.ndarray, U_true: np.ndarray) -> np.ndarray:
        """Relative L2 error at each time column."""
        assert U_pred.shape == U_true.shape
        T = U_pred.shape[1]
        errs = np.empty(T)
        for j in range(T):
            errs[j] = DMD.rel_l2(U_pred[:, j], U_true[:, j])
        return errs
