#!/usr/bin/env python3
"""
Potential Flow Theory for Orbital Shaking in a Cylinder

Computes the linear potential theory prediction for wall free-surface elevation
in an orbitally-shaken cylindrical container.

Reference: Canonical linear theory (Alpresa-style formulation)
"""

import numpy as np
from scipy.special import jn, jn_zeros
import csv


# Physical constants
G = 9.81  # m/s^2


def compute_natural_frequencies(R, d, n_modes=30):
    """
    Compute natural frequencies ω_1n for m=1 sloshing modes.
    
    Parameters:
    -----------
    R : float
        Cylinder radius (m)
    d : float
        Liquid depth at rest (m)
    n_modes : int
        Number of modes to compute
        
    Returns:
    --------
    omega_n : ndarray
        Natural frequencies (rad/s) for modes n=1,2,...,n_modes
    epsilon_n : ndarray
        Bessel roots ε_1n (zeros of J_1')
    """
    # Get zeros of J_1' (derivative of J_1)
    # These are roots of J_0 (since d/dx[J_1(x)] = J_0(x) - J_1(x)/x)
    # For large x, J_1'(x) ≈ -J_0(x), so we use J_0 zeros as approximation
    # More precisely: zeros of J_1' are between consecutive zeros of J_0 and J_2
    
    # Use known first few values, then approximate
    epsilon_exact = np.array([1.8412, 5.3314, 8.5363, 11.7060, 14.8636])
    
    if n_modes <= len(epsilon_exact):
        epsilon_n = epsilon_exact[:n_modes]
    else:
        # Extend with approximation for higher modes
        epsilon_n = np.zeros(n_modes)
        epsilon_n[:len(epsilon_exact)] = epsilon_exact
        # Asymptotic formula: ε_1n ≈ (n + 1/4)π for large n
        for n in range(len(epsilon_exact), n_modes):
            epsilon_n[n] = (n + 1 + 0.25) * np.pi
    
    # Compute λ_1n = ε_1n / R
    lambda_n = epsilon_n / R
    
    # Dimensionless depth parameter
    Gamma = d / R
    
    # Natural frequencies: ω_1n² = g λ_1n tanh(λ_1n d)
    omega_n = np.sqrt(G * lambda_n * np.tanh(lambda_n * d))
    
    return omega_n, epsilon_n


def compute_wall_amplitude(R, a, omega, d, n_modes=30):
    """
    Compute predicted wall amplitude A_PT using linear potential theory.
    
    Parameters:
    -----------
    R : float
        Cylinder radius (m)
    a : float
        Orbital radius (m)
    omega : float
        Forcing frequency (rad/s)
    d : float
        Liquid depth at rest (m)
    n_modes : int
        Number of modes in series
        
    Returns:
    --------
    A_PT : float
        Predicted wall amplitude (m)
    F : float
        Froude number (dimensionless)
    """
    # Compute Froude number
    F = a * omega**2 / G
    
    # Get natural frequencies
    omega_n, epsilon_n = compute_natural_frequencies(R, d, n_modes)
    
    # Compute series sum
    series_sum = 0.0
    for n in range(n_modes):
        eps = epsilon_n[n]
        omega_ratio_sq = (omega_n[n] / omega)**2
        
        # Avoid division by zero at resonance
        if abs(omega_ratio_sq - 1.0) < 1e-6:
            print(f"Warning: Near-resonance detected at mode n={n+1}")
            continue
            
        term = 1.0 / ((eps**2 - 1) * (omega_ratio_sq - 1))
        series_sum += term
    
    # Wall amplitude
    A_PT = 2 * R * F * (1 + series_sum)
    
    return A_PT, F


def generate_wall_elevation_csv(case_dir, R, a, freq, d, 
                                 duration=10.0, dt=0.01, 
                                 n_theta=64, n_modes=30,
                                 output_file=None):
    """
    Generate CSV with potential flow theory prediction for wall elevation.
    
    Parameters:
    -----------
    case_dir : str
        Path to case directory
    R : float
        Cylinder radius (m)
    a : float
        Orbital radius (m)
    freq : float
        Forcing frequency (Hz)
    d : float
        Liquid depth at rest (m)
    duration : float
        Simulation duration (s)
    dt : float
        Time step for output (s)
    n_theta : int
        Number of angular bins
    n_modes : int
        Number of modes in PT series
    output_file : str, optional
        Output CSV filename (default: case_dir/potential_flow_wall.csv)
        
    Returns:
    --------
    str : Path to output CSV file
    dict : Summary statistics
    """
    import os
    
    if output_file is None:
        output_file = os.path.join(case_dir, "potential_flow_wall.csv")
    
    # Convert frequency to rad/s
    omega = 2 * np.pi * freq
    
    # Compute wall amplitude
    A_PT, F = compute_wall_amplitude(R, a, omega, d, n_modes)
    Delta_h_PT = 2 * A_PT
    
    # Generate time array
    t_array = np.arange(0, duration + dt, dt)
    
    # Generate theta array (0 to 2π)
    theta_array = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    
    # Write CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'theta', 'zeta_wall'])
        
        for t in t_array:
            for theta in theta_array:
                # ζ_w(θ,t) = A_PT * cos(ωt - θ)
                zeta = A_PT * np.cos(omega * t - theta)
                writer.writerow([f'{t:.6f}', f'{theta:.6f}', f'{zeta:.8e}'])
    
    # Summary statistics
    summary = {
        'A_PT': A_PT,
        'Delta_h_PT': Delta_h_PT,
        'F': F,
        'R': R,
        'a': a,
        'omega': omega,
        'freq': freq,
        'd': d,
        'n_modes': n_modes,
        'output_file': output_file
    }
    
    return output_file, summary


def print_summary(summary):
    """Print formatted summary of PT prediction."""
    print("\n" + "="*60)
    print("  Potential Flow Theory Prediction")
    print("="*60)
    print(f"  Cylinder Radius (R):       {summary['R']:.4f} m")
    print(f"  Orbital Radius (a):        {summary['a']:.4f} m")
    print(f"  Forcing Frequency (f):     {summary['freq']:.4f} Hz")
    print(f"  Angular Frequency (ω):     {summary['omega']:.4f} rad/s")
    print(f"  Liquid Depth (d):          {summary['d']:.4f} m")
    print(f"  Froude Number (F):         {summary['F']:.6f}")
    print("-"*60)
    print(f"  Wall Amplitude (A_PT):     {summary['A_PT']:.6e} m")
    print(f"  Crest-to-Trough (Δh_PT):   {summary['Delta_h_PT']:.6e} m")
    print("-"*60)
    print(f"  Modes Used:                {summary['n_modes']}")
    print(f"  Output File:               {summary['output_file']}")
    print("="*60 + "\n")


def generate_video_from_csv(csv_file, case_dir, R, duration=10.0, fps=30):
    """
    Generate MP4 video from potential flow CSV data.
    
    Parameters:
    -----------
    csv_file : str
        Path to CSV file with potential flow data
    case_dir : str
        Case directory for output
    R : float
        Cylinder radius (for visualization scaling)
    duration : float
        Video duration (s)
    fps : int
        Frames per second
        
    Returns:
    --------
    str : Path to output video file
    """
    import os
    import numpy as np
    import csv
    
    # Check if matplotlib is available
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, FFMpegWriter
    except ImportError:
        print("  ⚠️  matplotlib not installed. Cannot generate video.")
        return None
    
    # Read CSV data
    times = []
    thetas = []
    zetas = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row['time']))
            thetas.append(float(row['theta']))
            zetas.append(float(row['zeta_wall']))
    
    # Convert to numpy arrays and reshape
    times = np.array(times)
    thetas = np.array(thetas)
    zetas = np.array(zetas)
    
    # Get unique times and thetas
    unique_times = np.unique(times)
    unique_thetas = np.unique(thetas)
    n_theta = len(unique_thetas)
    
    # Reshape data
    zeta_grid = zetas.reshape(-1, n_theta)
    
    # Setup figure
    fig = plt.figure(figsize=(12, 10))
    
    # 3D surface plot
    ax1 = fig.add_subplot(2, 2, (1, 3), projection='3d')
    
    # 2D unwrapped view
    ax2 = fig.add_subplot(2, 2, 2)
    
    # Time series at specific angles
    ax3 = fig.add_subplot(2, 2, 4)
    
    # Convert to Cartesian for 3D plot
    def polar_to_cart(theta, r, z):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y, z
    
    # Animation update function
    def update(frame):
        ax1.clear()
        ax2.clear()
        ax3.clear()
        
        t_idx = int(frame * len(unique_times) / (fps * duration))
        if t_idx >= len(unique_times):
            t_idx = len(unique_times) - 1
        
        t = unique_times[t_idx]
        zeta_t = zeta_grid[t_idx, :]
        
        # 3D surface
        theta_mesh = np.linspace(0, 2*np.pi, 100)
        zeta_interp = np.interp(theta_mesh, unique_thetas, zeta_t, period=2*np.pi)
        
        x, y, z = polar_to_cart(theta_mesh, R, zeta_interp)
        
        # Plot cylinder wall
        theta_cyl = np.linspace(0, 2*np.pi, 50)
        z_cyl = np.linspace(zeta_interp.min(), zeta_interp.max(), 20)
        Theta_cyl, Z_cyl = np.meshgrid(theta_cyl, z_cyl)
        X_cyl = R * np.cos(Theta_cyl)
        Y_cyl = R * np.sin(Theta_cyl)
        ax1.plot_surface(X_cyl, Y_cyl, Z_cyl, alpha=0.1, color='gray')
        
        # Plot interface
        ax1.plot(x, y, z, 'b-', linewidth=3, label='Interface')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('ζ (m)')
        ax1.set_title(f'Potential Flow Theory\nt = {t:.3f} s')
        ax1.set_box_aspect([1, 1, 0.5])
        
        # 2D unwrapped view
        ax2.plot(unique_thetas, zeta_t, 'b-', linewidth=2)
        ax2.set_xlabel('θ (rad)')
        ax2.set_ylabel('ζ (m)')
        ax2.set_title('Unwrapped Interface')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 2*np.pi])
        
        # Time series
        angles_to_plot = [0, np.pi/2, np.pi]
        for angle in angles_to_plot:
            idx = np.argmin(np.abs(unique_thetas - angle))
            zeta_series = zeta_grid[:t_idx+1, idx]
            ax3.plot(unique_times[:t_idx+1], zeta_series, 
                    label=f'θ = {angle:.2f}', linewidth=2)
        
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('ζ (m)')
        ax3.set_title('Interface Height vs Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim([0, duration])
        
        plt.tight_layout()
    
    # Create animation
    n_frames = int(fps * duration)
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps)
    
    # Save video
    output_file = os.path.join(case_dir, "potential_flow_animation.mp4")
    writer = FFMpegWriter(fps=fps, bitrate=2000)
    
    try:
        anim.save(output_file, writer=writer)
        plt.close(fig)
        return output_file
    except Exception as e:
        print(f"  ⚠️  Error saving video: {e}")
        print(f"  (FFmpeg may not be installed)")
        plt.close(fig)
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate potential flow theory prediction for orbital shaking"
    )
    parser.add_argument("case_dir", help="Case directory")
    parser.add_argument("--R", type=float, required=True, help="Cylinder radius (m)")
    parser.add_argument("--a", type=float, required=True, help="Orbital radius (m)")
    parser.add_argument("--freq", type=float, required=True, help="Frequency (Hz)")
    parser.add_argument("--d", type=float, required=True, help="Liquid depth (m)")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration (s)")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step (s)")
    parser.add_argument("--n-theta", type=int, default=64, help="Angular bins")
    parser.add_argument("--n-modes", type=int, default=30, help="Number of modes")
    parser.add_argument("--output", help="Output CSV file")
    
    args = parser.parse_args()
    
    output_file, summary = generate_wall_elevation_csv(
        args.case_dir, args.R, args.a, args.freq, args.d,
        args.duration, args.dt, args.n_theta, args.n_modes,
        args.output
    )
    
    print_summary(summary)
