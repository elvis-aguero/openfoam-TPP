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
    Generate MP4 video from potential flow CSV data, showing moving cylinder.
    """
    import os
    import numpy as np
    import csv
    
    # Check if matplotlib is available
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, FFMpegWriter
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("  ⚠️  matplotlib not installed. Cannot generate video.")
        return None
    
    # Extract params from summary/filename if possible, else default
    # Needed for motion: a (orbital radius), omega (freq)
    # We'll try to get them from summary dict passed in usually, but here we just have CSV.
    # Hack: Parse filename or just assume from input args if available. 
    # Actually, main.py passes these args to generate_wall_elevation_csv, 
    # but generate_video_from_csv only gets csv_file, case_dir, R, duration.
    # Let's verify we can get 'a' and 'omega' from somewhere.
    # For now, we'll try to re-parse from case_dir name.
    
    import re
    match = re.match(r'case_H([\d.]+)_D([\d.]+)_(\w+)_R([\d.]+)_f([\d.]+)', os.path.basename(case_dir))
    if match:
        a = float(match.group(4)) # R_orbital
        f_hz = float(match.group(5))
        omega = 2 * np.pi * f_hz
    else:
        a = 0.005 # Default if fail
        omega = 1.0
        
    # Read CSV data
    times = []
    thetas = []
    zetas = [] # Wall elevation
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row['time']))
            thetas.append(float(row['theta']))
            zetas.append(float(row['zeta_wall']))
    
    times = np.array(times)
    thetas = np.array(thetas)
    zetas = np.array(zetas)
    
    unique_times = np.unique(times)
    unique_thetas = np.unique(thetas)
    n_theta = len(unique_thetas)
    zeta_grid = zetas.reshape(-1, n_theta)
    
    # Setup Figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Pre-calculate cylinder mesh (unit radius, then scale)
    theta_mesh = np.linspace(0, 2*np.pi, 50)
    z_mesh = np.linspace(-0.1, 0.1, 10) # Placeholder height
    THETA, Z_wall = np.meshgrid(theta_mesh, z_mesh)
    X_cyl_unit = np.cos(THETA)
    Y_cyl_unit = np.sin(THETA)
    
    # Grid for liquid surface
    r_surface = np.linspace(0, R, 20)
    theta_surface = np.linspace(0, 2*np.pi, 50)
    R_surf, TH_surf = np.meshgrid(r_surface, theta_surface)
    X_surf_rel = R_surf * np.cos(TH_surf)
    Y_surf_rel = R_surf * np.sin(TH_surf)
    
    # To properly reconstruct the full surface eta(r,theta), we technically need the Bessel modes.
    # The CSV only has wall elevation. 
    # APPROXIMATION for Visualization: 
    # Assume the fundamental mode dominates: eta(r, theta) ~ J1(k*r)*cos(theta).
    # Wall elevation is at r=R. So eta(r) = eta(R) * J1(k*r)/J1(k*R).
    # Since we don't have coefficients, a linear radial profile r/R is a "decent" visual proxy 
    # for the sloshing mode, or we can just flat interpolate from 0 at center? 
    # Actually mode 1 is antisymmetric, so center is 0. 
    # We will scale the wall elevation by (r/R) for a visual "sloshing" effect.
    
    def update(frame):
        ax.clear()
        
        t_idx = int(frame * len(unique_times) / (fps * duration))
        if t_idx >= len(unique_times): t_idx = len(unique_times) - 1
        t = unique_times[t_idx]
        
        # 1. Cylinder Motion (Orbital)
        # x_c(t) = a * cos(omega * t)
        # y_c(t) = a * sin(omega * t)
        # Verify phase with your solver. Usually standard orbital is counter-clockwise.
        xc = a * np.cos(omega * t)
        yc = a * np.sin(omega * t)
        
        # 2. Update Cylinder Wall Position
        X_wall = R * X_cyl_unit + xc
        Y_wall = R * Y_cyl_unit + yc
        # Use wall elevation limits to set cylinder height dynamic or fixed
        z_min = -R # Arbitrary visual depth
        z_max = +R 
        # But we want the wall to look like a container.
        # Let's fix it relative to a mean level z=0.
        ax.plot_surface(X_wall, Y_wall, Z_wall*0 + z_max, color='k', alpha=0.1) # Top rim
        ax.plot_surface(X_wall, Y_wall, Z_wall*0 + z_min, color='k', alpha=0.1) # Bottom rim
        # Vertical struts or wireframe
        for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
            ax.plot([R*np.cos(angle)+xc, R*np.cos(angle)+xc], 
                    [R*np.sin(angle)+yc, R*np.sin(angle)+yc], 
                    [z_min, z_max], 'k-', lw=0.5, alpha=0.3)
            
        # 3. Update Surface
        # Get wall elevation at this time
        zeta_wall_t = zeta_grid[t_idx, :] # vector over theta
        # Interpolate to surface grid theta
        zeta_wall_interp = np.interp(theta_surface, unique_thetas, zeta_wall_t, period=2*np.pi)
        # Tile for radial 
        # zeta(r, theta) approx = zeta(R, theta) * (r/R)
        Z_surf = np.outer(zeta_wall_interp, r_surface / R)
        
        X_surf = X_surf_rel + xc
        Y_surf = Y_surf_rel + yc
        
        # Plot Surface
        surf = ax.plot_surface(X_surf, Y_surf, Z_surf, cmap='coolwarm', 
                               linewidth=0, antialiased=False, alpha=0.8)
        
        # Settings
        ax.set_xlim([-R-a, R+a])
        ax.set_ylim([-R-a, R+a])
        ax.set_zlim([-R/2, R/2])
        ax.set_xlabel('X')
        ax.set_title(f'Potential Flow Prediction\nt={t:.2f} s')
        
        # Ground reference text
        ax.text2D(0.05, 0.95, "Moving Mesh Frame", transform=ax.transAxes)

    n_frames = int(fps * duration)
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps)
    
    output_file = os.path.join(case_dir, "postProcessing", "potential_flow", "potential_flow_animation.mp4")
    
    # Try to find ffmpeg
    try:
        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        matplotlib.rcParams['animation.ffmpeg_path'] = ffmpeg_path
    except:
        pass
        
    writer = FFMpegWriter(fps=fps, bitrate=2000)
    try:
        anim.save(output_file, writer=writer)
        plt.close(fig)
        return output_file
    except Exception as e:
        print(f"Error saving video: {e}")
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
