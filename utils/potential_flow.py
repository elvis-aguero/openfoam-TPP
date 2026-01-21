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
import os


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
        # Default to generic name in current dir if not specified
        output_file = "potential_flow_wall.csv"
        if os.path.isdir(case_dir): 
             output_file = os.path.join(case_dir, output_file)
    
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


def generate_video_from_csv(csv_file, output_dir, R, duration=10.0, fps=30):
    """
    Wrapper to generate both 3D and Dashboard videos from potential flow CSV.
    Saves videos directly to output_dir.
    """
    print(f"    - Generating 3D moving mesh animation in {output_dir}...")
    generate_3d_animation(csv_file, output_dir, R, duration, fps)
    
    print(f"    - Generating analysis dashboard animation in {output_dir}...")
    generate_dashboard_animation(csv_file, output_dir, R, duration, fps)
    
    return output_dir

def generate_3d_animation(csv_file, output_dir, R, duration, fps):
    import os
    import numpy as np
    import csv
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, FFMpegWriter
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        return None

    # Parse parameters (a, omega)
    a = 0.005; omega = 1.0 # Defaults
    
    # Try to parse from filename
    import re
    match = re.search(r'_H([\d.]+)_D([\d.]+)_(\w+)_R([\d.]+)_f([\d.]+)', csv_file)
    if match:
        a = float(match.group(4)) 
        f_hz = float(match.group(5))
        omega = 2 * np.pi * f_hz
    elif os.path.basename(output_dir).startswith("case_"):
         match = re.search(r'R([\d.]+)_f([\d.]+)', os.path.basename(output_dir))
         if match:
            a = float(match.group(1))
            f_hz = float(match.group(2))
            omega = 2 * np.pi * f_hz

    # Load Data
    times, thetas, zetas = [], [], []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row['time']))
            thetas.append(float(row['theta']))
            zetas.append(float(row['zeta_wall']))
    
    unique_times = np.unique(times)
    unique_thetas = np.unique(thetas)
    # Reshape: (n_times, n_thetas)
    n_t, n_th = len(unique_times), len(unique_thetas)
    if len(zetas) == n_t * n_th:
       zeta_wall_grid = np.array(zetas).reshape(n_t, n_th)
    else:
       print("Warning: Grid mismatch in CSV. Reshaping might fail.")
       zeta_wall_grid = np.zeros((n_t, n_th))

    # Debug: Check Amplitude
    max_amp = np.max(np.abs(zeta_wall_grid))
    print(f"      3D Animation Debug: Max Wall Amplitude = {max_amp:.6e} m")

    # Setup 3D Figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1) 

    # Pre-calc Coordinate Systems
    # 1. Cylinder Coordinates
    theta_cyl = np.linspace(0, 2*np.pi, 60)
    z_cyl = np.linspace(-R, R/2, 2)
    THETA, Z_wall_cyl = np.meshgrid(theta_cyl, z_cyl)
    X_unit, Y_unit = np.cos(THETA), np.sin(THETA)
    
    # 2. Surface Grid (Polar)
    r_surf = np.linspace(0, R, 20)
    # Use CSV thetas for compatibility
    R_grid, Th_grid = np.meshgrid(r_surf, unique_thetas) # (n_th, n_r)
    X_surf_rel = R_grid * np.cos(Th_grid)
    Y_surf_rel = R_grid * np.sin(Th_grid)
    
    # Bessel Function First Mode shape for radial dependence J1(eps*r/R)
    # First root of J1' is approx 1.8412
    # We import jn from scipy.special at top level
    eps1 = 1.8412
    J1_r = jn(1, eps1 * r_surf / R)
    J1_R = jn(1, eps1) # normalization at wall
    Radial_Profile = J1_r / J1_R  # Shape (n_r,) normalized to 1 at r=R

    def update(frame):
        ax.clear()
        t_idx = int(frame * len(unique_times) / (fps * duration))
        if t_idx >= n_t: t_idx = n_t - 1
        
        t = unique_times[t_idx]
        # Tank Motion
        xc, yc = a * np.cos(omega * t), a * np.sin(omega * t)
        
        # Plot Tank Wireframe
        ax.plot_surface(R*X_unit + xc, R*Y_unit + yc, Z_wall_cyl, color='k', alpha=0.05)
        ax.plot_wireframe(R*X_unit + xc, R*Y_unit + yc, Z_wall_cyl, color='k', alpha=0.1, rstride=1, cstride=10)

        # Plot Free Surface
        # Wall Elevation at this time: zeta_wall(theta) -> (n_th,)
        zw_t = zeta_wall_grid[t_idx, :]
        
        # Reconstruct full surface Z(r, theta) using separation of variables approx:
        # Z(r, theta) = zeta_wall(theta) * (J1(k*r)/J1(k*R))
        # zw_t is (n_th,), Radial_Profile is (n_r,)
        Z_surf = zw_t[:, np.newaxis] * Radial_Profile[np.newaxis, :]
        
        ax.plot_surface(X_surf_rel + xc, Y_surf_rel + yc, Z_surf, cmap='coolwarm', 
                              alpha=0.9, rstride=1, cstride=1, linewidth=0, antialiased=False)
        
        view_r = R * 1.5
        ax.set_box_aspect([1,1,0.5])
        ax.set_xlim([-view_r, view_r]); ax.set_ylim([-view_r, view_r]); ax.set_zlim([-R/2, R/2])
        ax.set_title(f"Potential Flow 3D (t={t:.2f}s)\nMax Z: {np.max(Z_surf):.4f}m", fontsize=10)
        ax.axis('off')

    run_animation_save(fig, update, duration, fps, output_dir, "potential_flow_3d.mp4")

def generate_dashboard_animation(csv_file, output_dir, R, duration, fps):
    import os
    import numpy as np
    import csv
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    # Load Data (Redundant but safe)
    times, thetas, zetas = [], [], []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row['time']))
            thetas.append(float(row['theta']))
            zetas.append(float(row['zeta_wall']))
    
    unique_times = np.unique(times)
    unique_thetas = np.unique(thetas)
    zeta_grid = np.array(zetas).reshape(len(unique_times), len(unique_thetas))

    # Setup Dashboard Figure
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1) # Unwrapped
    ax2 = fig.add_subplot(1, 2, 2) # Time Series
    
    def update(frame):
        ax1.clear(); ax2.clear()
        t_idx = int(frame * len(unique_times) / (fps * duration))
        if t_idx >= len(unique_times): t_idx = len(unique_times) - 1
        
        # 1. Unwrapped View
        zeta_t = zeta_grid[t_idx, :]
        ax1.plot(unique_thetas, zeta_t, 'b-', lw=2)
        ax1.set_xlabel('Theta (rad)'); ax1.set_ylabel('Elevation (m)')
        ax1.set_title(f'Wall Elevation Profile (t={unique_times[t_idx]:.2f}s)')
        ax1.grid(True)
        ax1.set_ylim([np.min(zeta_grid), np.max(zeta_grid)])
        
        # 2. Time Series (Probes at 0, 90, 180 deg)
        for ang_deg, color in zip([0, 90, 180], ['r', 'g', 'b']):
            ang_rad = np.deg2rad(ang_deg)
            idx = np.abs(unique_thetas - ang_rad).argmin()
            ax2.plot(unique_times[:t_idx], zeta_grid[:t_idx, idx], color=color, label=f'{ang_deg}°')
            
        ax2.set_xlim([0, duration])
        ax2.set_ylim([np.min(zeta_grid), np.max(zeta_grid)])
        ax2.legend(loc='upper right')
        ax2.set_xlabel('Time (s)'); ax2.set_title('Wave Probes')
        ax2.grid(True)
        
    run_animation_save(fig, update, duration, fps, output_dir, "potential_flow_dashboard.mp4")

def run_animation_save(fig, update_func, duration, fps, output_dir, filename):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    import os
    
    n_frames = int(fps * duration)
    anim = FuncAnimation(fig, update_func, frames=n_frames, interval=1000/fps)
    
    # Save directly to output_dir
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    
    # FFMPEG Setup
    try:
        import imageio_ffmpeg
        plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
    except: pass
    
    writer = FFMpegWriter(fps=fps, bitrate=2000)
    try:
        anim.save(out_path, writer=writer)
        print(f"      ✅ Saved: {filename}")
    except Exception as e:
        print(f"      ❌ Warning: Failed to save {filename}: {e}")
    plt.close(fig)


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
