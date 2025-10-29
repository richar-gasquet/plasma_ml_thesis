#!/usr/bin/env python3
"""
Simple Extended Physics-Informed DMD for Nonlinear Diffusion

This script demonstrates the key concepts of Extended Physics-Informed DMD
for the nonlinear diffusion equation with a focus on clarity and understanding.

The nonlinear equation: ∂u/∂t = ∂/∂x[D(x) * ∂u/∂x] + S(x)
where D(x) = ν_min + ν₀ * u²

Key insight: Include the nonlinear term z = ∂/∂x[D(x) * ∂u/∂x] in the state vector.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dmd_utils import DMD

# Nonlinear diffusion parameters (must match solver)
NU_MIN = 1e-4
NU_0 = 1e-2

def face_gradients(u, dx):
    """Compute face-centered gradients: g_{i+1/2} = (u_{i+1} - u_i)/dx"""
    return (u[1:] - u[:-1]) / dx

def compute_nonlinear_term(u, dx):
    """
    Compute z = ∂/∂x[D(x) * ∂u/∂x] where D(x) = ν_min + ν₀ * u²
    
    This is the key function that captures the nonlinear physics.
    """
    # Compute face-centered gradients
    g = face_gradients(u, dx)
    
    # Compute face-centered u values and diffusion coefficient
    u_faces = 0.5 * (u[1:] + u[:-1])  # Face-centered u
    D = NU_MIN + NU_0 * (u_faces**2)  # Nonlinear diffusion coefficient
    
    # Compute flux F = -D * ∂u/∂x
    F = -D * g
    
    # Compute divergence of flux: ∂/∂x[F] = (F_{i+1/2} - F_{i-1/2})/dx
    z = np.zeros_like(u)
    z[1:-1] = -(F[1:] - F[:-1]) / dx  # Interior points
    
    # Boundary conditions: z = 0 at boundaries
    z[0] = 0.0
    z[-1] = 0.0
    
    return z

def run_simple_comparison():
    """Run a simple comparison between standard and extended DMD."""
    
    # Load data
    print("Loading nonlinear diffusion data...")
    x = np.loadtxt("diffusion_bounded_nonlinear/x.txt")
    dt = float(np.loadtxt("diffusion_bounded_nonlinear/dt.txt"))
    U = np.loadtxt("diffusion_bounded_nonlinear/U.txt")
    S = np.loadtxt("diffusion_bounded_nonlinear/U_source.txt")
    
    # Ensure correct shape
    if U.shape[0] != x.shape[0]:
        U = U.T
    
    N, T = U.shape
    dx = float(x[1] - x[0])
    
    print(f"Data loaded: N={N}, T={T}, dt={dt:.6e}")
    
    # Parameters
    K = 1000  # Training steps
    H = 1000  # Forecast steps
    
    # 1. Standard DMD with [u; S; 1]
    print("\n=== Standard DMD (will fail) ===")
    
    # Prepare standard augmented state
    S_broadcast = np.tile(S[:, None], (1, T)) if S.ndim == 1 else S
    ones_row = np.ones((1, T))
    U_standard = np.vstack([U, S_broadcast, ones_row])
    
    # Train standard DMD
    dmd_standard = DMD(r=None, energy_thresh=0.999, tlsq=0)
    res_standard = dmd_standard.fit(U_standard, dt, n_train=K)
    
    # Forecast with standard injection
    x_prev = U_standard[:, K-1].copy()
    x_prev[N:2*N] = S_broadcast[:, K-1]
    x_prev[-1] = 1.0
    
    preds_standard = np.zeros((U_standard.shape[0], H))
    for j in range(H):
        step = dmd_standard.forecast(1, x_init=x_prev)
        xj = step[:, -1] if step.ndim > 1 else step
        
        # Inject exact S and bias
        idx = min(K + j, T - 1)
        xj[N:2*N] = S_broadcast[:, idx]
        xj[-1] = 1.0
        
        preds_standard[:, j] = xj
        x_prev = xj
    
    U_pred_standard = np.hstack([res_standard.Uhat_train[:N, :], preds_standard[:N, :]])
    
    # 2. Extended DMD with [u; S; 1; z]
    print("\n=== Extended Physics-Informed DMD ===")
    
    # Compute nonlinear terms for all time steps
    print("Computing nonlinear terms...")
    Z = np.zeros_like(U)
    for j in range(T):
        Z[:, j] = compute_nonlinear_term(U[:, j], dx)
        if (j + 1) % 500 == 0:
            print(f"  Computed {j+1}/{T} nonlinear terms")
    
    # Prepare extended augmented state
    U_extended = np.vstack([U, S_broadcast, ones_row, Z])
    
    # Train extended DMD
    dmd_extended = DMD(r=None, energy_thresh=0.999, tlsq=0)
    res_extended = dmd_extended.fit(U_extended, dt, n_train=K)
    
    # Forecast with extended injection
    x_prev = U_extended[:, K-1].copy()
    x_prev[N:2*N] = S_broadcast[:, K-1]
    x_prev[-1] = 1.0
    x_prev[2*N+1:] = Z[:, K-1]
    
    preds_extended = np.zeros((U_extended.shape[0], H))
    for j in range(H):
        step = dmd_extended.forecast(1, x_init=x_prev)
        xj = step[:, -1] if step.ndim > 1 else step
        
        # Inject exact S, bias, and nonlinear term
        idx = min(K + j, T - 1)
        xj[N:2*N] = S_broadcast[:, idx]
        xj[-1] = 1.0
        xj[2*N+1:] = Z[:, idx]
        
        preds_extended[:, j] = xj
        x_prev = xj
    
    U_pred_extended = np.hstack([res_extended.Uhat_train[:N, :], preds_extended[:N, :]])
    
    # 3. Analysis and visualization
    print("\n=== Analysis ===")
    
    # Compute errors
    T_eval = min(T, K + H)
    U_true_eval = U[:, :T_eval]
    U_pred_standard_eval = U_pred_standard[:, :T_eval]
    U_pred_extended_eval = U_pred_extended[:, :T_eval]
    
    rel_err_standard = DMD.rel_l2_over_time(U_pred_standard_eval, U_true_eval)
    rel_err_extended = DMD.rel_l2_over_time(U_pred_extended_eval, U_true_eval)
    
    print(f"Standard DMD - Mean error: {np.mean(rel_err_standard):.2e}")
    print(f"Extended DMD - Mean error: {np.mean(rel_err_extended):.2e}")
    print(f"Improvement factor: {np.mean(rel_err_standard) / np.mean(rel_err_extended):.1f}x")
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Solution comparison at different times
    times_to_plot = [K//4, K//2, K-1, K+H//2, K+H-1]
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    
    for i, t_idx in enumerate(times_to_plot):
        if t_idx < T_eval:
            ax1.plot(x, U_true_eval[:, t_idx], color=colors[i], alpha=0.7, linewidth=2, 
                    label=f'True t={t_idx*dt:.3f}' if i < 3 else "")
            ax1.plot(x, U_pred_standard_eval[:, t_idx], color=colors[i], alpha=0.7, 
                    linewidth=1, linestyle='--', label=f'Standard t={t_idx*dt:.3f}' if i < 3 else "")
            ax1.plot(x, U_pred_extended_eval[:, t_idx], color=colors[i], alpha=0.7, 
                    linewidth=1, linestyle=':', label=f'Extended t={t_idx*dt:.3f}' if i < 3 else "")
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('u(x,t)')
    ax1.set_title('Solution Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error evolution
    t_eval = dt * np.arange(T_eval)
    ax2.plot(t_eval, rel_err_standard, 'r-', linewidth=2, label='Standard DMD')
    ax2.plot(t_eval, rel_err_extended, 'g-', linewidth=2, label='Extended DMD')
    ax2.axvline(x=t_eval[K-1], color='black', linestyle='--', alpha=0.7, label='Training End')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Relative L2 Error')
    ax2.set_title('Error Evolution')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Nonlinear term
    mid_idx = T // 2
    ax3.plot(x, Z[:, mid_idx], 'k-', linewidth=2, label=f'z(x,t) at t={mid_idx*dt:.3f}')
    ax3.set_xlabel('x')
    ax3.set_ylabel('z = ∂/∂x[D(x)∂u/∂x]')
    ax3.set_title('Nonlinear Term')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Performance comparison
    metrics = ['Mean', 'Median', 'Max', 'Final']
    standard_vals = [np.mean(rel_err_standard), np.median(rel_err_standard), 
                    np.max(rel_err_standard), rel_err_standard[-1]]
    extended_vals = [np.mean(rel_err_extended), np.median(rel_err_extended), 
                    np.max(rel_err_extended), rel_err_extended[-1]]
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    ax4.bar(x_pos - width/2, standard_vals, width, label='Standard DMD', color='red', alpha=0.7)
    ax4.bar(x_pos + width/2, extended_vals, width, label='Extended DMD', color='green', alpha=0.7)
    ax4.set_xlabel('Error Metrics')
    ax4.set_ylabel('Relative L2 Error')
    ax4.set_title('Performance Comparison')
    ax4.set_yscale('log')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nonlinear_dmd_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print(f"\n=== SUMMARY ===")
    print(f"Standard DMD rank: {res_standard.r}")
    print(f"Extended DMD rank: {res_extended.r}")
    print(f"Nonlinear term shape: {Z.shape}")
    print(f"Extended state dimension: {U_extended.shape[0]} (vs {U_standard.shape[0]} for standard)")
    print(f"Mean error improvement: {np.mean(rel_err_standard) / np.mean(rel_err_extended):.1f}x")
    print(f"Final error improvement: {rel_err_standard[-1] / rel_err_extended[-1]:.1f}x")
    
    return U_pred_standard, U_pred_extended, Z, rel_err_standard, rel_err_extended

if __name__ == "__main__":
    run_simple_comparison()
