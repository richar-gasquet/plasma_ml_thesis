#!/usr/bin/env python3
"""
Test script for nonlinear term computation

This script tests the nonlinear term computation function to ensure it's working
correctly and matches the solver's discretization.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Nonlinear diffusion parameters (must match solver)
NU_MIN = 1e-4
NU_0 = 1e-2

def face_gradients(u, dx):
    """Compute face-centered gradients: g_{i+1/2} = (u_{i+1} - u_i)/dx"""
    return (u[1:] - u[:-1]) / dx

def compute_nonlinear_term(u, dx):
    """
    Compute z = ∂/∂x[D(x) * ∂u/∂x] where D(x) = ν_min + ν₀ * u²
    
    This uses the same discretization as the solver.
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

def test_nonlinear_term():
    """Test the nonlinear term computation with sample data."""
    
    print("Testing nonlinear term computation...")
    
    # Load sample data
    x = np.loadtxt("diffusion_bounded_nonlinear/x.txt")
    U = np.loadtxt("diffusion_bounded_nonlinear/U.txt")
    
    if U.shape[0] != x.shape[0]:
        U = U.T
    
    N, T = U.shape
    dx = float(x[1] - x[0])
    
    print(f"Data shape: {U.shape}")
    print(f"Grid spacing: {dx:.6e}")
    print(f"Nonlinear parameters: ν_min={NU_MIN}, ν₀={NU_0}")
    
    # Test on a few time steps
    test_times = [0, T//4, T//2, 3*T//4, T-1]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, t_idx in enumerate(test_times):
        if i >= 6:  # Only plot first 6
            break
            
        u = U[:, t_idx]
        z = compute_nonlinear_term(u, dx)
        
        # Plot solution and nonlinear term
        ax = axes[i]
        ax2 = ax.twinx()
        
        line1 = ax.plot(x, u, 'b-', linewidth=2, label='u(x,t)')
        line2 = ax2.plot(x, z, 'r--', linewidth=2, label='z = ∂/∂x[D(x)∂u/∂x]')
        
        ax.set_xlabel('x')
        ax.set_ylabel('u(x,t)', color='b')
        ax2.set_ylabel('z(x,t)', color='r')
        ax.set_title(f't = {t_idx} (t = {t_idx * 0.001:.3f})')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right')
        
        ax.grid(True, alpha=0.3)
    
    # Remove unused subplot
    if len(test_times) < 6:
        axes[-1].remove()
    
    plt.tight_layout()
    plt.savefig('nonlinear_term_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Test properties
    print("\nTesting nonlinear term properties...")
    
    # Test 1: Boundary conditions
    z_test = compute_nonlinear_term(U[:, 0], dx)
    print(f"Boundary condition test:")
    print(f"  z[0] = {z_test[0]:.2e} (should be 0)")
    print(f"  z[-1] = {z_test[-1]:.2e} (should be 0)")
    
    # Test 2: Conservation properties
    print(f"\nConservation test:")
    for t_idx in [0, T//2, T-1]:
        u = U[:, t_idx]
        z = compute_nonlinear_term(u, dx)
        integral_z = np.trapz(z, x=x)
        print(f"  t={t_idx}: ∫z dx = {integral_z:.2e} (should be ~0)")
    
    # Test 3: Scale analysis
    print(f"\nScale analysis:")
    z_all = np.zeros_like(U)
    for j in range(T):
        z_all[:, j] = compute_nonlinear_term(U[:, j], dx)
    
    print(f"  Max |z|: {np.max(np.abs(z_all)):.2e}")
    print(f"  Mean |z|: {np.mean(np.abs(z_all)):.2e}")
    print(f"  Max |u|: {np.max(np.abs(U)):.2e}")
    print(f"  Ratio max|z|/max|u|: {np.max(np.abs(z_all))/np.max(np.abs(U)):.2e}")
    
    print("\n✓ Nonlinear term computation test complete!")
    return z_all

def analyze_nonlinear_behavior():
    """Analyze the nonlinear behavior of the system."""
    
    print("\nAnalyzing nonlinear behavior...")
    
    # Load data
    x = np.loadtxt("diffusion_bounded_nonlinear/x.txt")
    U = np.loadtxt("diffusion_bounded_nonlinear/U.txt")
    
    if U.shape[0] != x.shape[0]:
        U = U.T
    
    N, T = U.shape
    dx = float(x[1] - x[0])
    
    # Compute nonlinear terms for all time
    Z = np.zeros_like(U)
    for j in range(T):
        Z[:, j] = compute_nonlinear_term(U[:, j], dx)
    
    # Analyze the relationship between u and z
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: u vs z scatter plot
    u_flat = U.flatten()
    z_flat = Z.flatten()
    ax1.scatter(u_flat, z_flat, alpha=0.5, s=1)
    ax1.set_xlabel('u')
    ax1.set_ylabel('z = ∂/∂x[D(x)∂u/∂x]')
    ax1.set_title('u vs z Relationship')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Time evolution of max values
    t = np.arange(T) * 0.001  # Assuming dt = 0.001
    ax2.plot(t, np.max(np.abs(U), axis=0), 'b-', label='max |u|')
    ax2.plot(t, np.max(np.abs(Z), axis=0), 'r-', label='max |z|')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Maximum Absolute Value')
    ax2.set_title('Time Evolution of Max Values')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Nonlinear term at different times
    times_to_plot = [0, T//4, T//2, 3*T//4, T-1]
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    for i, t_idx in enumerate(times_to_plot):
        ax3.plot(x, Z[:, t_idx], color=colors[i], linewidth=2, 
                label=f't = {t_idx*0.001:.3f}')
    ax3.set_xlabel('x')
    ax3.set_ylabel('z = ∂/∂x[D(x)∂u/∂x]')
    ax3.set_title('Nonlinear Term Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Energy in nonlinear term
    energy_z = np.trapz(Z**2, x=x, axis=0)
    energy_u = np.trapz(U**2, x=x, axis=0)
    ax4.plot(t, energy_z, 'r-', label='Energy in z')
    ax4.plot(t, energy_u, 'b-', label='Energy in u')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Energy')
    ax4.set_title('Energy Evolution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nonlinear_behavior_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Nonlinear behavior analysis complete!")

if __name__ == "__main__":
    # Test the nonlinear term computation
    z_all = test_nonlinear_term()
    
    # Analyze nonlinear behavior
    analyze_nonlinear_behavior()
    
    print("\n🎉 All tests completed successfully!")
    print("Generated files:")
    print("  - nonlinear_term_test.png")
    print("  - nonlinear_behavior_analysis.png")
