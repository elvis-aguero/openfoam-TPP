#!/usr/bin/env python3
import os
import sys
import shutil
import subprocess
import argparse

# --- Dependency Management ---
def ensure_dependencies():
    """Check and install required Python packages."""
    # Check if we have already tried to restart in this process tree
    if os.environ.get("SLOSHING_ENV_RESTARTED") == "1":
        in_venv = True
    else:
        # Fallback check
        venv_path = os.path.join(os.path.dirname(__file__), "sloshing")
        in_venv = venv_path in sys.executable
    
    try:
        import numpy
        import scipy
        import matplotlib
        import pyvista
        import imageio
        import imageio_ffmpeg
        import h5py
    except ImportError as e:
        if in_venv:
            print(f"\n⚠️  Dependency missing in virtual environment: {e}")
            print("Attempting to repair environment by re-installing requirements...")
            # Proceed to install logic below instead of exiting
            venv_path = os.path.join(os.path.dirname(__file__), "sloshing") # Re-define for safety
        else:
            print(f"\n⚠️  Missing dependencies detected: {e}")
            venv_path = os.path.join(os.path.dirname(__file__), "sloshing")
            print(f"Creating virtual environment: {venv_path}")
            subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
        
        # Get pip from venv
        if sys.platform == "win32":
            pip_path = os.path.join(venv_path, "Scripts", "pip")
            python_path = os.path.join(venv_path, "Scripts", "python")
        else:
            pip_path = os.path.join(venv_path, "bin", "pip")
            python_path = os.path.join(venv_path, "bin", "python3")
        
        # Install requirements
        req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
        subprocess.run([pip_path, "install", "-r", req_file], check=True)
        
        print("\n✅ Dependencies installed successfully!")
        print(f"Restarting with virtual environment...\n")
        
        # Set environment variable to prevent loop
        os.environ["SLOSHING_ENV_RESTARTED"] = "1"
        
        # Restart script with venv python
        os.execv(python_path, [python_path] + sys.argv)
    except Exception as e:
        print(f"\n❌ Error during dependency import: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Run dependency check
ensure_dependencies()

import math
import itertools
import re

# --- Constants & Defaults ---
TEMPLATE_DIR = "circularSloshingTank"
DEFAULTS = {
    "H": 0.1,
    "D": 0.02,
    "mesh": 0.002,
    "geo": "flat",
    "R": 0.003,
    "freq": 2.0,
    "duration": 10.0,
    "dt": 0.001,
    "ramp": -1,
    "n_cpus": 1,
}

# --- Utility Functions ---

def parse_range(s):
    """
    Parses a MATLAB-style range (start:step:end) or comma-separated list.
    Returns a list of floats.
    """
    s = s.strip()
    if ':' in s:
        parts = s.split(':')
        if len(parts) == 2:
            start, end = float(parts[0]), float(parts[1])
            step = 1.0
        elif len(parts) == 3:
            start, step, end = float(parts[0]), float(parts[1]), float(parts[2])
        else:
            raise ValueError(f"Invalid range format: {s}")
        # Generate range
        vals = []
        v = start
        while v <= end + 1e-9:  # Tolerance for floating point
            vals.append(round(v, 6))
            v += step
        return vals
    else:
        # Comma-separated
        return [float(x.strip()) for x in s.split(',')]

def parse_indices(s, max_idx):
    """
    Parses comma-separated indices and ranges (e.g., "1, 3-5, 7").
    Returns a list of 0-indexed integers.
    """
    indices = set()
    for part in s.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            for i in range(int(start), int(end) + 1):
                if 1 <= i <= max_idx:
                    indices.add(i - 1)
        else:
            i = int(part)
            if 1 <= i <= max_idx:
                indices.add(i - 1)
    return sorted(list(indices))

def get_case_name(params):
    """Generates a unique case folder name from parameters."""
    return f"case_H{params['H']}_D{params['D']}_{params['geo']}_R{params['R']}_f{params['freq']}_d{params['duration']}_m{params['mesh']}"

def is_case_done(case_dir, duration):
    """Checks if the simulation for this case is complete."""
    # Check if final time folder exists with alpha.water
    final_time_str = str(int(duration)) if duration == int(duration) else str(duration)
    final_path = os.path.join(case_dir, final_time_str, "alpha.water")
    return os.path.exists(final_path)

def parse_case_params(case_name):
    """Extracts parameters from a case folder name."""
    # Format: case_H{H}_D{D}_{geo}_R{R}_f{freq}_d{duration}_m{mesh}
    match = re.match(r'case_H([\d.]+)_D([\d.]+)_(\w+)_R([\d.]+)_f([\d.]+)_d([\d.]+)_m([\d.]+)', case_name)
    if not match:
        return DEFAULTS.copy()
    
    return {
        "H": float(match.group(1)),
        "D": float(match.group(2)),
        "geo": match.group(3),
        "R": float(match.group(4)),
        "freq": float(match.group(5)),
        "duration": float(match.group(6)),
        "mesh": float(match.group(7)),
        "dt": DEFAULTS['dt'],
        "ramp": DEFAULTS['ramp']
    }

def estimate_resources(params):
    """
    Estimates required CPUs, memory, and wall-clock time.
    Model: ~160 cpu-hours per 1M cells per 1s simulation.
    """
    h, d, mesh_size = params['H'], params['D'], params['mesh']
    duration = params['duration']
    
    vol = math.pi * ((d / 2.0)**2) * h
    cell_vol = mesh_size ** 3
    n_cells = vol / cell_vol
    
    # Calibrated performance model
    # User's recent runs suggest high CPU usage per cell.
    # Increasing safety factor significantly to prevent timeouts.
    # Factor: 80.0 cpu-hours per (Mcell-sec)
    total_cpu_hours = (n_cells / 1e6) * duration * 80.0
    
    # Suggest CPUs (aim for ~4-8 hours wall-clock time)
    suggested_cpus = math.ceil(total_cpu_hours / 6.0)
    
    # --- Efficiency Guard ---
    # Don't over-parallelize. OpenFOAM sweet spot is 20k-50k cells/core.
    # Minimum 15k cells per core to avoid communication bottlenecks.
    max_efficient_cpus = max(1, int(n_cells / 15000))
    suggested_cpus = min(suggested_cpus, max_efficient_cpus)
    
    # Cap at 32 for Oscar free tier / general stability
    suggested_cpus = min(suggested_cpus, 32)
    
    # For power-of-two enthusiasts or scotch efficiency
    if suggested_cpus > 1:
        # Round to nearest power of 2 for better decomposition balance
        suggested_cpus = 2**math.floor(math.log2(suggested_cpus))

    wall_clock_hours = total_cpu_hours / suggested_cpus
    
    # Add 50% buffer + 1 hour flat
    safe_hours = wall_clock_hours * 1.5 + 1.0
    
    # Enforce realistic minimums for Oscar
    # If case is obviously tiny, 1h is fine. If larger, force 4h+.
    if n_cells > 100000:
        safe_hours = max(safe_hours, 6.0)
    else:
        safe_hours = max(safe_hours, 1.0)
        
    # Cap at 24h to avoid long queue times (unless absolutely needed)
    safe_hours = min(safe_hours, 24.0)
    
    time_limit = format_time(safe_hours)
    
    # Memory: 200MB per 100k cells + base
    mem_gb = (n_cells / 100000) * 0.2 + 2.0
    mem_gb = max(4.0, math.ceil(mem_gb))
    
    return f"{int(mem_gb)}G", time_limit, n_cells, suggested_cpus
    # Add 100% buffer (2x) to account for variability & I/O
    wall_clock_hours *= 2.0
    
    # Format for Slurm
    h_str = f"{int(wall_clock_hours):02d}"
    m_str = f"{int((wall_clock_hours % 1) * 60):02d}"
    time_limit = f"{h_str}:{m_str}:00"
    
    # Memory: ~2GB per 100k cells
    mem_gb = math.ceil((n_cells / 1e5) * 2.0)
    mem_gb = max(8, min(mem_gb, 128))
    
    return f"{mem_gb}G", time_limit, n_cells, suggested_cpus

# --- Core Actions ---

def setup_case(params):
    """Creates the case directory and runs setup scripts."""
    case_name = get_case_name(params)
    
    if os.path.exists(case_name):
        print(f"  ⚠️  {case_name} already exists. Skipping.")
        return case_name
    
    print(f"  📂 Creating: {case_name}")
    shutil.copytree(TEMPLATE_DIR, case_name)
    
    # Ensure writable
    for root, dirs, files in os.walk(case_name):
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o777)
        for f in files:
            os.chmod(os.path.join(root, f), 0o666)

    cwd = os.path.join(os.getcwd(), case_name)
    
    # Motion
    subprocess.run([
        sys.executable, "generate_motion.py", 
        str(params['R']), str(params['freq']), str(params['duration']), 
        str(params['dt']), str(params['ramp'])
    ], cwd=cwd, check=True, capture_output=True)
    
    # Fields
    subprocess.run([sys.executable, "update_setFields.py", str(params['H'])], 
                   cwd=cwd, check=True, capture_output=True)
    
    # Mesh Geometry
    subprocess.run([
        sys.executable, "generate_mesh.py", 
        str(params['H']), str(params['D']), str(params['mesh']), params['geo']
    ], cwd=cwd, check=True, capture_output=True)
    
    # Run Gmsh
    gmsh_path = shutil.which("gmsh")
    if gmsh_path:
        subprocess.run([
            "gmsh", "-3", "cylinder.geo", "-format", "msh2", "-o", "cylinder.msh"
        ], cwd=cwd, check=True, capture_output=True)
    else:
        print("  ❌ gmsh not found in PATH. Cannot generate mesh.")

    # Parallel Setup (Inject numberOfSubdomains)
    if params.get('n_cpus', 1) > 1:
        decomp_path = os.path.join(cwd, "system", "decomposeParDict")
        if os.path.exists(decomp_path):
            with open(decomp_path, 'r') as f:
                content = f.read()
            content = re.sub(r'numberOfSubdomains\s+\d+;', f'numberOfSubdomains {params["n_cpus"]};', content)
            with open(decomp_path, 'w') as f:
                f.write(content)

    # Update controlDict endTime
    control_path = os.path.join(cwd, "system", "controlDict")
    if os.path.exists(control_path):
        with open(control_path, 'r') as f:
            content = f.read()
        content = re.sub(r'endTime\s+[\d.]+;', f'endTime {params["duration"]};', content)
        with open(control_path, 'w') as f:
            f.write(content)
        
    return case_name

def run_case_local(case_name, n_cpus=1):
    """Runs simulation locally."""
    # Check for existing progress
    has_progress = os.path.isdir(os.path.join(case_name, "processor0"))
    if not has_progress:
        # Check for serial time folders (excluding '0')
        time_folders = [d for d in os.listdir(case_name) if d.replace('.','',1).isdigit() and d != '0']
        if time_folders:
            has_progress = True
            
    if has_progress:
        print(f"  🏃 Resuming {case_name} (CPUs={n_cpus})...")
        subprocess.run(["make", "-C", case_name, "resume", f"N_CPUS={n_cpus}"], check=True)
    else:
        print(f"  🏃 Running {case_name} (CPUs={n_cpus})...")
        subprocess.run(["make", "-C", case_name, "run", f"N_CPUS={n_cpus}"], check=True)

def run_case_oscar(case_name, params, is_oscar):
    """Submits job to Slurm on Oscar."""
    mem, time_limit, n_cells, _ = estimate_resources(params)
    
    # Read the ACTUAL number of subdomains from the case folder
    # This is the single source of truth for parallel runs
    n_cpus = 1
    decomp_path = os.path.join(case_name, "system", "decomposeParDict")
    if os.path.exists(decomp_path):
        with open(decomp_path, 'r') as f:
            content = f.read()
            match = re.search(r'numberOfSubdomains\s+(\d+);', content)
            if match:
                n_cpus = int(match.group(1))

    script_path = os.path.join(case_name, "run_simulation.slurm")
    
    header = [
        "#!/usr/bin/env bash",
        f"#SBATCH -J {case_name}",
        "#SBATCH -p batch",
        "#SBATCH -N 1",
        f"#SBATCH -n {n_cpus}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH -o {case_name}/slurm.%j.out",
        f"#SBATCH -e {case_name}/slurm.%j.err",
        "#SBATCH --mail-type=END",
        "#SBATCH --mail-user=elvis_vera@brown.edu",
        "",
        "set -euo pipefail",
        "export OMP_NUM_THREADS=1",
        "",
        f"echo 'Case: {case_name}'",
        "# Check if we are resuming (parallel processors or serial time folders existed)",
        "if [ -d 'processor0' ] || (ls -d [0-9]* 2>/dev/null | grep -v '^0$' | grep -q .) ; then",
        "    echo 'Found existing progress. Resuming simulation...'",
        f"    make -C {case_name} resume OSCAR=1 N_CPUS={n_cpus}",
        "else",
        "    echo 'Starting fresh simulation...'",
        f"    make -C {case_name} run OSCAR=1 N_CPUS={n_cpus}",
        "fi",
        "echo 'End: $(date)'"
    ]
    
    with open(script_path, "w") as f:
        f.write("\n".join(header))
    
    print(f"  🚀 Submitting {case_name} ({n_cpus} CPUs, {mem}, {time_limit})...")
    subprocess.run(["sbatch", script_path], check=True)

# --- Menu System ---

# Human-readable labels for parameters
PARAM_LABELS = {
    "H": "Height (m)",
    "D": "Diameter (m)",
    "mesh": "Mesh Size (m)",
    "geo": "Geometry",
    "R": "Motion Radius (m)",
    "freq": "Motion Frequency (Hz)",
    "duration": "Duration (s)",
    "dt": "Time Step (s)",
    "ramp": "Soft Start Ramp (s, -1=auto)",
    "n_cpus": "Parallel CPUs (1=serial)",
}

GEO_OPTIONS = ["flat", "cap"]

def display_config(current_values, sweeps):
    """Displays the current configuration with any overrides."""
    print("\nCurrent Configuration:")
    param_keys = list(DEFAULTS.keys())
    for i, k in enumerate(param_keys):
        label = PARAM_LABELS.get(k, k)
        if k in sweeps:
            val_str = str(sweeps[k])
            print(f"  {i+1}) {label:25}: {val_str} (SWEEP)")
        else:
            print(f"  {i+1}) {label:25}: {current_values[k]}")

def menu_build_cases(is_oscar):
    """Submenu 1: Build Case Setups"""
    print("\n--- Build Case Setups ---")
    
    current_values = DEFAULTS.copy()
    sweeps = {}
    param_keys = list(DEFAULTS.keys())
    
    while True:
        display_config(current_values, sweeps)
        print("\nOptions: Enter number to edit, 'done' to build, 'cancel' to abort.")
        
        user_input = input("Select: ").strip()
        
        if user_input.lower() == 'cancel':
            print("Cancelled.")
            return
        
        if user_input.lower() == 'done':
            break
        
        # Parse selection
        param = None
        if user_input.isdigit():
            idx = int(user_input) - 1
            if 0 <= idx < len(param_keys):
                param = param_keys[idx]
        else:
            match = [k for k in DEFAULTS if k.lower() == user_input.lower()]
            if match:
                param = match[0]
        
        if not param:
            print(f"  Invalid selection: {user_input}")
            continue
        
        # Special handling for 'geo' (categorical)
        if param == 'geo':
            print(f"\n  Select geometry:")
            for i, opt in enumerate(GEO_OPTIONS):
                print(f"    {i+1}) {opt}")
            geo_input = input("  Choice (or comma-separated for sweep, e.g., '1,2'): ").strip()
            try:
                if ',' in geo_input:
                    indices = [int(x.strip()) - 1 for x in geo_input.split(',')]
                    sweeps[param] = [GEO_OPTIONS[i] for i in indices]
                else:
                    idx = int(geo_input) - 1
                    current_values[param] = GEO_OPTIONS[idx]
                    if param in sweeps:
                        del sweeps[param]
            except (ValueError, IndexError):
                print("  Invalid choice.")
            continue
        
        # Numeric parameters
        label = PARAM_LABELS.get(param, param)
        val_str = input(f"  Enter value(s) for '{label}' (single or sweep, e.g., 0.1 or 0.1:0.05:0.2): ").strip()
        try:
            vals = parse_range(val_str)
            if len(vals) == 1:
                current_values[param] = vals[0]
                if param in sweeps:
                    del sweeps[param]
            else:
                sweeps[param] = vals
        except ValueError as e:
            print(f"  ❌ Error: {e}")
    
    # Confirmation
    display_config(current_values, sweeps)
    
    # Build param_sets
    if not sweeps:
        param_sets = [current_values.copy()]
    else:
        lengths = [len(v) for v in sweeps.values()]
        
        if len(set(lengths)) == 1:
            print(f"\n✅ All sweep lists are length {lengths[0]}. Using ZIP mode.")
            keys = list(sweeps.keys())
            param_sets = []
            for i in range(lengths[0]):
                p = current_values.copy()
                for k in keys:
                    p[k] = sweeps[k][i]
                param_sets.append(p)
        else:
            total = 1
            for l in lengths:
                total *= l
            confirm = input(f"\n⚠️  Sweep lists have different lengths. This will generate {total} cases (Cartesian Product). Continue? (y/n): ").strip().lower()
            if confirm != 'y':
                print("Cancelled.")
                return
            
            keys = list(sweeps.keys())
            combos = list(itertools.product(*[sweeps[k] for k in keys]))
            param_sets = []
            for combo in combos:
                p = current_values.copy()
                for i, k in enumerate(keys):
                    p[k] = combo[i]
                param_sets.append(p)
    
    # Final Case Review & Resource Estimation
    print("\n" + "="*40)
    print("   Final Review & Resource Estimation")
    print("="*40)
    
    # Calculate for the first case in param_sets to show representative estimate
    sample_params = param_sets[0]
    mem, time_limit, n_cells, suggested_cpus = estimate_resources(sample_params)
    
    print(f"Total Cases to Build: {len(param_sets)}")
    print(f"Estimated Cells per Case: {int(n_cells):,}")
    print(f"Suggested Wall-Clock Time: {time_limit}")
    print(f"Suggested Parallelization: {suggested_cpus} CPUs")
    
    if suggested_cpus > 1 and current_values['n_cpus'] == 1:
        print(f"\n💡 [RECOMMENDED] Multi-processing is highly recommended for this cell count.")
        use_multi = input(f"   Enable parallel execution with {suggested_cpus} CPUs? (y/n): ").strip().lower()
        if use_multi == 'y':
            for p in param_sets:
                p['n_cpus'] = suggested_cpus
    
    # Final confirmation
    confirm = input(f"\nConfirm building {len(param_sets)} case(s)? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return
    
    print(f"\nGenerating {len(param_sets)} case(s)...")
    for params in param_sets:
        setup_case(params)
    print("✅ Done building cases.")

def menu_run_cases(is_oscar):
    """Submenu 2: Run Cases"""
    print("\n--- Run Cases ---")
    
    cases = sorted([d for d in os.listdir('.') if os.path.isdir(d) and d.startswith('case_')])
    if not cases:
        print("No cases found. Use 'Build Case Setups' first.")
        return
    
    # Display cases with status
    print("Available Cases:")
    for i, c in enumerate(cases):
        # Try to infer duration from folder name (hacky, but works for now)
        # Or assume default
        status = "(DONE)" if is_case_done(c, DEFAULTS['duration']) else ""
        print(f"  {i+1}) {c} {status}")
    
    idx_str = input("\nEnter case indices to run (e.g., 1, 3-5, all): ").strip().lower()
    if idx_str == 'all':
        indices = list(range(len(cases)))
    else:
        indices = parse_indices(idx_str, len(cases))
    
    if not indices:
        print("No valid indices selected.")
        return
    
    print(f"\nRunning {len(indices)} case(s)...")
    
    has_openfoam = shutil.which("foamRun") is not None
    
    for i in indices:
        case_name = cases[i]
        params = parse_case_params(case_name)
        
        if is_oscar:
            run_case_oscar(case_name, params, is_oscar)
        elif has_openfoam:
            # Estimate resources to get n_cpus for local run
            _, _, _, n_cpus = estimate_resources(params)
            run_case_local(case_name, n_cpus=n_cpus)
        else:
            print(f"  ❌ OpenFOAM not installed. Cannot run {case_name} locally.")

def generate_video(case_dir):
    """Generates a video from OpenFOAM results using PyVista."""
    import pyvista as pv
    import imageio
    import numpy as np
    
    # Force off-screen rendering for cluster environment
    pv.OFF_SCREEN = True
    
    # Try to launch XVFB for headless rendering if available, but don't fail if not
    # This helps if X is technically present but no display is attached
    try:
        pv.start_xvfb()
    except OSError:
        pass

    print(f"  🎬 Generating video for {case_dir} using PyVista...")
    
    foam_file = os.path.join(case_dir, "case.foam")
    if not os.path.exists(foam_file):
        # Create empty .foam file if it doesn't exist (PyVista needs it)
        with open(foam_file, 'w') as f:
            pass
            
    try:
        reader = pv.POpenFOAMReader(foam_file)
        # Force reading cell data
        reader.cell_to_point_creation = False 
    except Exception as e:
         print(f"  ❌ Error loading OpenFOAM case: {e}")
         return False

    # Get Time Values
    try:
        time_values = reader.time_values
    except AttributeError:
        # Fallback if time_values not directly accessible
        time_values = reader.reader.GetTimeValues()
        
    print(f"  Found {len(time_values)} timesteps.")
    
    # Setup Output in CASE folder
    results_dir = os.path.join(case_dir, "postProcessing")
    os.makedirs(results_dir, exist_ok=True)

    # 1. Generate 3D Moving Mesh Video
    print("    - Generating 3D perspective video...")
    video_path_3d = os.path.join(results_dir, "animation_3d.mp4")
    
    # Setup Plotter (Off-screen)
    plotter = pv.Plotter(off_screen=True, window_size=[1280, 720])
    plotter.camera_position = [(0.0, -0.2, 0.15), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    
    try:
        with imageio.get_writer(video_path_3d, fps=30, macro_block_size=None) as writer:
            for i, t in enumerate(time_values):
                reader.set_active_time_value(t)
                mesh = reader.read()
                
                if mesh.n_blocks > 0:
                    internal_mesh = mesh[0]
                    plotter.clear()
                    
                    # 1. Plot Water Surface (Isosurface alpha.water = 0.5)
                    if 'alpha.water' in internal_mesh.cell_data:
                        mesh_point = internal_mesh.cell_data_to_point_data()
                        try:
                            isosurface = mesh_point.contour(isosurfaces=[0.5], scalars='alpha.water')
                            plotter.add_mesh(isosurface, color='deepskyblue', smooth_shading=True, 
                                           specular=0.5, show_edges=False, label='Water')
                        except: pass
                    
                    # 2. Plot Tank Container
                    plotter.add_mesh(internal_mesh.outline(), color='black', opacity=0.3)
                    # 3. Floor
                    plotter.add_floor(face='z', i_resolution=10, j_resolution=10, color='gray', pad=1.0)
                    plotter.add_text(f"OpenFOAM 3D\nTime: {t:.2f} s", position='upper_left', font_size=10, color='black')
                    
                    writer.append_data(plotter.screenshot(return_img=True))
                    
                if (i+1) % 20 == 0:
                     print(f"      Rendered 3D frame {i+1}/{len(time_values)}")
        print(f"      ✅ Saved: animation_3d.mp4")
    except Exception as e:
        print(f"      ❌ Error saving 3D video: {e}")

    # 2. Generate Dashboard Video
    # We use a helper from potential_flow to avoid duplicate code
    sys.path.insert(0, 'utils')
    try:
        from potential_flow import generate_dashboard_animation
        csv_path = os.path.join(results_dir, "interface", "wall_elevation.csv")
        if os.path.exists(csv_path):
            print("    - Generating dashboard analysis video...")
            # Detect R for plotting
            import re
            match = re.search(r'_D([\d.]+)_', os.path.basename(case_dir))
            R_val = float(match.group(1))/2.0 if match else 0.1
            
            generate_dashboard_animation(csv_path, case_dir, R_val, duration=time_values[-1], fps=30)
            # Find and rename the file generated by the helper
            dash_src = os.path.join(case_dir, "postProcessing", "potential_flow", "potential_flow_dashboard.mp4")
            dash_dst = os.path.join(results_dir, "animation_dashboard_openfoam.mp4")
            # Potential flow helper saves to potential_flow subfolder, we move it up
            if os.path.exists(dash_src):
                if os.path.exists(dash_dst): os.remove(dash_dst)
                os.rename(dash_src, dash_dst)
                print(f"      ✅ Saved: animation_dashboard_openfoam.mp4")
    except Exception as e:
        print(f"      ⚠️  Could not generate dashboard: {e}")

    return True
        
def extract_interface(case_dir):
    """Extracts the water-air interface (alpha.water=0.5) using PyVista."""
    import pyvista as pv
    import numpy as np
    
    print(f"  📊 Extracting interface for {case_dir} using PyVista...")
    
    foam_file = os.path.join(case_dir, "case.foam")
    if not os.path.exists(foam_file):
        with open(foam_file, 'w') as f:
            pass
            
    try:
        reader = pv.POpenFOAMReader(foam_file)
    except Exception as e:
         print(f"  ❌ Error loading OpenFOAM case: {e}")
         return False

    time_values = reader.time_values
    
    # Setup Output in CASE folder
    results_dir = os.path.join(case_dir, "postProcessing", "interface")
    os.makedirs(results_dir, exist_ok=True)
    
    csv_summary = ["time,max_z,min_z,mean_z,num_points"]
    csv_wall = ["time,theta,zeta_wall"] # For dashboard
    
    # Parse R from case name
    import re
    match = re.search(r'_D([\d.]+)_', os.path.basename(case_dir))
    R_target = float(match.group(1))/2.0 if match else 0.1

    print(f"  Processing {len(time_values)} timesteps (R={R_target})...")
    
    for i, t in enumerate(time_values):
        reader.set_active_time_value(t)
        mesh = reader.read()
        
        if mesh.n_blocks > 0:
            internal_mesh = mesh[0]
            if 'alpha.water' in internal_mesh.cell_data:
                mesh_point = internal_mesh.cell_data_to_point_data()
                try:
                    isosurface = mesh_point.contour(isosurfaces=[0.5], scalars='alpha.water')
                    
                    # Save VTP
                    vtp_file = os.path.join(results_dir, f'interface_t{t:.6f}.vtp')
                    isosurface.save(vtp_file)
                    
                    if isosurface.n_points > 0:
                        pts = isosurface.points
                        z_coords = pts[:, 2]
                        # Aggregate Stats
                        csv_summary.append(f"{t},{np.max(z_coords)},{np.min(z_coords)},{np.mean(z_coords)},{len(pts)}")
                        
                        # Extract Wall elevation profile for dashboard
                        # We project points to (r, theta) and pick points near r=R
                        r = np.sqrt(pts[:,0]**2 + pts[:,1]**2)
                        # Find points near the wall (within 2% margin)
                        wall_mask = r > (R_target * 0.98)
                        if np.any(wall_mask):
                            wall_pts = pts[wall_mask]
                            wall_thetas = np.arctan2(wall_pts[:,1], wall_pts[:,0])
                            # Bin by theta to get a clean profile
                            n_bins = 64
                            bins = np.linspace(-np.pi, np.pi, n_bins+1)
                            for b in range(n_bins):
                                bin_mask = (wall_thetas >= bins[b]) & (wall_thetas < bins[b+1])
                                if np.any(bin_mask):
                                    z_bin = np.mean(wall_pts[bin_mask, 2])
                                    theta_bin = (bins[b] + bins[b+1])/2.0
                                    csv_wall.append(f"{t},{theta_bin},{z_bin}")
                    else:
                        csv_summary.append(f"{t},0,0,0,0")
                except:
                    csv_summary.append(f"{t},0,0,0,0")
            else:
                csv_summary.append(f"{t},0,0,0,0")
        else:
            csv_summary.append(f"{t},0,0,0,0")
            
        if (i+1) % 20 == 0:
            print(f"    Processed {i+1}/{len(time_values)}")
            
    # Save CSVs
    with open(os.path.join(results_dir, 'interface_summary.csv'), 'w') as f:
        f.write('\n'.join(csv_summary))
    with open(os.path.join(results_dir, 'wall_elevation.csv'), 'w') as f:
        f.write('\n'.join(csv_wall))
        
    print(f"  ✅ Extraction complete.")
    return True

def generate_potential_flow(case_dir):
    """Generates potential flow theory prediction for wall elevation."""
    import sys
    sys.path.insert(0, 'utils')
    from potential_flow import generate_wall_elevation_csv, print_summary, generate_video_from_csv
    
    print(f"  📐 Generating potential flow prediction for {case_dir}...")
    
    # Parse parameters from case name
    # Format: case_H{H}_D{D}_{geo}_R{R}_f{freq}_d{duration}_m{mesh}
    import re
    match = re.match(r'case_H([\d.]+)_D([\d.]+)_(\w+)_R([\d.]+)_f([\d.]+)_d([\d.]+)_m([\d.]+)', case_dir)
    if not match:
        print(f"  ❌ Could not parse parameters from case name: {case_dir}")
        return False
    
    H = float(match.group(1))
    D = float(match.group(2))
    geo = match.group(3)
    R_orbital = float(match.group(4))
    freq = float(match.group(5))
    duration = float(match.group(6))
    mesh_size = float(match.group(7))
    
    # Cylinder radius
    R_cyl = D / 2.0
    
    # Liquid depth (assume H/2 fill level)
    d = H / 2.0
    
    # Duration (try to get from case, otherwise use default)
    duration = DEFAULTS['duration']
    dt = 0.01  # Output time step
    
    # Setup Output in CASE folder
    results_dir = os.path.join(case_dir, "postProcessing", "potential_flow")
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        # Generate CSV
        csv_file = os.path.join(results_dir, "potential_flow_wall.csv")
        output_file, summary = generate_wall_elevation_csv(
            case_dir, R_cyl, R_orbital, freq, d,
            duration=duration, dt=dt, n_theta=64, n_modes=30,
            output_file=csv_file
        )
        print_summary(summary)
        print(f"  ✅ Potential flow data saved: {output_file}")
        
        # Generate video
        print(f"  🎬 Generating potential flow animation...")
        video_file = generate_video_from_csv(output_file, results_dir, R_cyl, duration, fps=30)
        if video_file:
            print(f"  ✅ Animation saved: {video_file}")
        
        return True
    except Exception as e:
        print(f"  ❌ Error generating potential flow: {e}")
        import traceback
        traceback.print_exc()
        return False

def menu_postprocess(is_oscar):
    """Submenu 3: Postprocess"""
    print("\n" + "="*60)
    print("  POSTPROCESS MENU")
    print("="*60)
    
    cases = sorted([d for d in os.listdir('.') if os.path.isdir(d) and d.startswith('case_')])
    if not cases:
        print("No cases found.")
        return
    
    # Display cases
    print("\nAvailable Cases:")
    for i, c in enumerate(cases):
        status = "(DONE)" if is_case_done(c, DEFAULTS['duration']) else ""
        print(f"  {i+1}) {c} {status}")
    
    print("\n" + "-"*60)
    print("Select Action:")
    print("  1) Generate Videos (OpenFOAM)")
    print("  2) Extract Interface Data (OpenFOAM)")
    print("  3) Generate Potential Flow Theory Prediction")
    print("  Q) Back to Main Menu")
    print("-"*60)
    
    choice = input("\nAction: ").strip()
    
    if choice == '1':
        print("\n→ Video Generation (OpenFOAM Results)")
        idx_str = input("  Enter case numbers (e.g., 1, 3-5, all): ").strip().lower()
        if idx_str == 'all':
            indices = list(range(len(cases)))
        else:
            indices = parse_indices(idx_str, len(cases))
        
        if not indices:
            print("No valid indices selected.")
            return
        
        print(f"\nGenerating videos for {len(indices)} case(s)...")
        for i in indices:
            if is_oscar:
                if i == indices[0]:
                    submit = input("\n⚠️  Heavy video rendering detected. Submit as Slurm job? (y/n): ").strip().lower()
                    if submit == 'y':
                        for idx in indices:
                            run_postprocess_oscar(cases[idx], "video")
                        return
            generate_video(cases[i])
    elif choice == '2':
        print("\n→ Interface Extraction (OpenFOAM Results)")
        idx_str = input("  Enter case numbers (e.g., 1, 3-5, all): ").strip().lower()
        if idx_str == 'all':
            indices = list(range(len(cases)))
        else:
            indices = parse_indices(idx_str, len(cases))
        
        if not indices:
            print("No valid indices selected.")
            return
        
        print(f"\nExtracting interfaces for {len(indices)} case(s)...")
        for i in indices:
            if is_oscar:
                if i == indices[0]:
                    submit = input("\n⚠️  Heavy post-processing detected. Submit as Slurm job? (y/n): ").strip().lower()
                    if submit == 'y':
                        for idx in indices:
                            run_postprocess_oscar(cases[idx], "interface")
                        return
            extract_interface(cases[i])
    elif choice == '3':
        print("\n→ Potential Flow Theory Prediction")
        idx_str = input("  Enter case numbers (e.g., 1, 3-5, all): ").strip().lower()
        if idx_str == 'all':
            indices = list(range(len(cases)))
        else:
            indices = parse_indices(idx_str, len(cases))
        
        if not indices:
            print("No valid indices selected.")
            return

        print(f"\nGenerating potential flow predictions for {len(indices)} case(s)...")
        for i in indices:
            if is_oscar:
                if i == indices[0]:
                    submit = input("\n⚠️  Animation rendering detected. Submit as Slurm job? (y/n): ").strip().lower()
                    if submit == 'y':
                        for idx in indices:
                            run_postprocess_oscar(cases[idx], "flow")
                        return
            generate_potential_flow(cases[i])

    elif choice == '4':
        print("\n→ Reconstruct Parallel Case (Merge Processor Directories)")
        idx_str = input("  Enter case numbers (e.g., 1, 3-5, all): ").strip().lower()
        if idx_str == 'all':
            indices = list(range(len(cases)))
        else:
            indices = parse_indices(idx_str, len(cases))
        
        if not indices:
            print("No valid indices selected.")
            return

        for i in indices:
            case = cases[i]
            print(f"\n  🔨 Reconstructing {case}...")
            # Check if processor dirs exist
            if not os.path.exists(os.path.join(case, "processor0")):
                print(f"     ⚠️ No processor0 directory found. Is this a parallel run?")
                continue
            
            if is_oscar:
                submit = input(f"     Submit reconstruction for {case} as Slurm job? (y/n/all): ").strip().lower()
                if submit == 'y' or submit == 'all':
                    run_postprocess_oscar(case, "reconstruct")
                    if submit == 'all':
                        # Silently submit the rest
                        for future_idx in indices[indices.index(i)+1:]:
                             run_postprocess_oscar(cases[future_idx], "reconstruct")
                        return
                elif submit == 'n':
                   subprocess.run(["reconstructPar"], cwd=case, check=False)
            else:
                 subprocess.run(["reconstructPar"], cwd=case, check=False)

    elif choice == '5':
        return

def run_postprocess_oscar(case_name, action):
    """Submits a post-processing job to Slurm."""
    script_path = os.path.join(case_name, f"postprocess_{action}.slurm")
    
    header = [
        "#!/usr/bin/env bash",
        f"#SBATCH -J post_{action}_{case_name}",
        "#SBATCH -p batch",
        "#SBATCH -N 1",
        "#SBATCH -n 1",
        "#SBATCH --time=01:00:00",
        "#SBATCH --mem=8G",
        f"#SBATCH -o {case_name}/postProcessing/slurm_postprocessing.log",
        "#SBATCH --open-mode=append",
        "",
        "set -euo pipefail",
        "",
        "echo '------------------------------------------------------------'",
        f"echo 'Action: {action} | Case: {case_name}'",
        f"echo 'Date: $(date)'",
        f"python3 main.py --headless --case {case_name} --action {action}",
        "echo 'End: $(date)'",
        "echo '------------------------------------------------------------'",
        ""
    ]
    
    os.makedirs(os.path.join(case_name, "postProcessing"), exist_ok=True)
    
    with open(script_path, "w") as f:
        f.write("\n".join(header))
    
    print(f"  🚀 Submitting post-processing job for {case_name} ({action})...")
    subprocess.run(["sbatch", script_path], check=True)

def main_menu():
    """Main entry point."""
    print("\n" + "="*40)
    print("   Sloshing Tank Manager")
    print("="*40)
    
    oscar_input = input("Are you on Oscar? (y/n): ").strip().lower()
    is_oscar = oscar_input == 'y'
    
    while True:
        print("\n--- Main Menu ---")
        print("1) Build Case Setups")
        print("2) Run Cases")
        print("3) Postprocess Cases")
        print("Q) Quit")
        
        choice = input("\nSelect an option: ").strip().lower()
        
        if choice == '1':
            menu_build_cases(is_oscar)
        elif choice == '2':
            menu_run_cases(is_oscar)
        elif choice == '3':
            menu_postprocess(is_oscar)
        elif choice == 'q':
            print("Goodbye!")
            break
        else:
            print("Invalid option.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Run without menu")
    parser.add_argument("--case", type=str, help="Case directory for headless mode")
    parser.add_argument("--action", type=str, choices=["video", "interface", "flow"], help="Action for headless mode")
    
    args = parser.parse_args()
    
    if args.headless:
        if not args.case or not args.action:
            print("Error: --case and --action are required in headless mode.")
            sys.exit(1)
        
        if args.action == "video":
            generate_video(args.case)
        elif args.action == "interface":
            extract_interface(args.case)
        elif args.action == "flow":
            generate_potential_flow(args.case)
    else:
        main_menu()
