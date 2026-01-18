#!/usr/bin/env python3
import os
import sys
import shutil
import subprocess
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
    return f"case_H{params['H']}_D{params['D']}_{params['geo']}_R{params['R']}_f{params['freq']}"

def is_case_done(case_dir, duration):
    """Checks if the simulation for this case is complete."""
    # Check if final time folder exists with alpha.water
    final_time_str = str(int(duration)) if duration == int(duration) else str(duration)
    final_path = os.path.join(case_dir, final_time_str, "alpha.water")
    return os.path.exists(final_path)

def estimate_resources(h, d, mesh_size):
    """Estimates required memory and time based on domain volume."""
    vol = math.pi * ((d / 2.0)**2) * h
    cell_vol = mesh_size ** 3
    n_cells = vol / cell_vol
    
    if n_cells < 100000:
        return "8G", "24:00:00", n_cells
    elif n_cells < 1000000:
        return "32G", "48:00:00", n_cells
    else:
        return "64G", "72:00:00", n_cells

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
        
    return case_name

def run_case_local(case_name):
    """Runs simulation locally."""
    print(f"  🏃 Running {case_name}...")
    subprocess.run(["make", "-C", case_name, "run"], check=True)

def run_case_oscar(case_name, params, is_oscar):
    """Submits job to Slurm on Oscar."""
    mem, time_limit, n_cells = estimate_resources(params['H'], params['D'], params['mesh'])
    script_path = os.path.join(case_name, "run_simulation.slurm")
    
    header = [
        "#!/usr/bin/env bash",
        f"#SBATCH -J {case_name}",
        "#SBATCH -p batch",
        "#SBATCH -N 1",
        "#SBATCH -n 1",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH -o {case_name}/slurm.%j.out",
        f"#SBATCH -e {case_name}/slurm.%j.err",
        "",
        "set -euo pipefail",
        "export OMP_NUM_THREADS=1",
        "",
        f"echo 'Case: {case_name}'",
        f"make -C {case_name} run OSCAR=1",
        "echo 'End: $(date)'"
    ]
    
    with open(script_path, "w") as f:
        f.write("\n".join(header))
    
    print(f"  🚀 Submitting {case_name} ({mem}, {time_limit})...")
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
    
    # Final confirmation
    confirm = input(f"\nBuild {len(param_sets)} case(s)? (y/n): ").strip().lower()
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
        params = DEFAULTS.copy()  # Ideally, parse from folder name or config
        
        if is_oscar:
            run_case_oscar(case_name, params, is_oscar)
        elif has_openfoam:
            run_case_local(case_name)
        else:
            print(f"  ❌ OpenFOAM not installed. Cannot run {case_name} locally.")

def generate_video(case_dir):
    """Generates a video from OpenFOAM results using ParaView."""
    print(f"  🎬 Generating video for {case_dir}...")
    
    # Create ParaView Python script
    script_content = f"""
from paraview.simple import *

# Load case
case = OpenFOAMReader(FileName='{case_dir}/case.foam')
case.MeshRegions = ['internalMesh']
case.CellArrays = ['alpha.water', 'U', 'p_rgh']

# Get animation scene
animationScene = GetAnimationScene()
animationScene.UpdateAnimationUsingDataTimeSteps()

# Create render view
renderView = CreateView('RenderView')
renderView.ViewSize = [1920, 1080]
renderView.Background = [1, 1, 1]

# Show data
display = Show(case, renderView)
display.Representation = 'Surface'
display.ColorArrayName = ['CELLS', 'alpha.water']

# Color by alpha.water
ColorBy(display, ('CELLS', 'alpha.water'))
alphaLUT = GetColorTransferFunction('alpha.water')
alphaLUT.RescaleTransferFunction(0.0, 1.0)
alphaLUT.ApplyPreset('Cool to Warm', True)

# Camera setup
renderView.ResetCamera()
renderView.CameraPosition = [0.0, -0.15, 0.1]
renderView.CameraFocalPoint = [0.0, 0.0, 0.05]
renderView.CameraViewUp = [0.0, 0.0, 1.0]

# Save animation
SaveAnimation('{case_dir}/animation.mp4', renderView, 
              ImageResolution=[1920, 1080],
              FrameRate=30)

print("Video saved to {case_dir}/animation.mp4")
"""
    
    script_path = os.path.join(case_dir, "render_video.py")
    with open(script_path, "w") as f:
        f.write(script_content)
    
    # Run pvpython
    pvpython = shutil.which("pvpython")
    if not pvpython:
        print(f"  ❌ pvpython not found. Install ParaView to generate videos.")
        return False
    
    try:
        subprocess.run([pvpython, script_path], cwd=case_dir, check=True, 
                      capture_output=True, text=True)
        print(f"  ✅ Video saved: {case_dir}/animation.mp4")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Error generating video: {e.stderr}")
        return False

def extract_interface(case_dir):
    """Extracts the water-air interface (alpha.water=0.5) for all timesteps."""
    print(f"  📊 Extracting interface for {case_dir}...")
    
    # Create ParaView Python script
    script_content = f"""
from paraview.simple import *
import os

# Load case
case = OpenFOAMReader(FileName='{case_dir}/case.foam')
case.MeshRegions = ['internalMesh']
case.CellArrays = ['alpha.water']

# Get timesteps
timesteps = case.TimestepValues
print(f"Found {{len(timesteps)}} timesteps")

# Create output directory
output_dir = '{case_dir}/interface_data'
os.makedirs(output_dir, exist_ok=True)

# Extract iso-surface at alpha.water = 0.5
contour = Contour(Input=case)
contour.ContourBy = ['CELLS', 'alpha.water']
contour.Isosurfaces = [0.5]

# CSV for summary statistics
csv_data = []
csv_data.append("time,max_z,min_z,mean_z,num_points")

for i, t in enumerate(timesteps):
    # Update to current timestep
    UpdatePipeline(time=t)
    
    # Save VTP file (full 3D interface)
    vtp_file = os.path.join(output_dir, f'interface_t{{t:.6f}}.vtp')
    SaveData(vtp_file, contour)
    
    # Get interface data for statistics
    data = servermanager.Fetch(contour)
    if data.GetNumberOfPoints() > 0:
        points = data.GetPoints()
        z_coords = [points.GetPoint(j)[2] for j in range(data.GetNumberOfPoints())]
        max_z = max(z_coords)
        min_z = min(z_coords)
        mean_z = sum(z_coords) / len(z_coords)
        num_pts = len(z_coords)
    else:
        max_z = min_z = mean_z = 0.0
        num_pts = 0
    
    csv_data.append(f"{{t}},{{max_z}},{{min_z}},{{mean_z}},{{num_pts}}")
    
    if (i+1) % 10 == 0:
        print(f"  Processed {{i+1}}/{{len(timesteps)}} timesteps")

# Save CSV summary
csv_file = os.path.join(output_dir, 'interface_summary.csv')
with open(csv_file, 'w') as f:
    f.write('\\n'.join(csv_data))

print(f"Interface data saved to {{output_dir}}/")
print(f"  - VTP files: interface_t*.vtp ({{len(timesteps)}} files)")
print(f"  - Summary: interface_summary.csv")
"""
    
    script_path = os.path.join(case_dir, "extract_interface.py")
    with open(script_path, "w") as f:
        f.write(script_content)
    
    # Run pvpython
    pvpython = shutil.which("pvpython")
    if not pvpython:
        print(f"  ❌ pvpython not found. Install ParaView to extract interfaces.")
        return False
    
    try:
        result = subprocess.run([pvpython, script_path], cwd=case_dir, check=True, 
                               capture_output=True, text=True)
        print(result.stdout)
        print(f"  ✅ Interface extracted: {case_dir}/interface_data/")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Error extracting interface: {e.stderr}")
        return False

def menu_postprocess(is_oscar):
    """Submenu 3: Postprocess"""
    print("\n--- Postprocess Cases ---")
    
    cases = sorted([d for d in os.listdir('.') if os.path.isdir(d) and d.startswith('case_')])
    if not cases:
        print("No cases found.")
        return
    
    # Display cases
    print("Available Cases:")
    for i, c in enumerate(cases):
        status = "(DONE)" if is_case_done(c, DEFAULTS['duration']) else ""
        print(f"  {i+1}) {c} {status}")
    
    print("\nPostprocess Options:")
    print("1) Generate Videos")
    print("2) Extract Interface Data")
    print("Q) Back")
    
    choice = input("\nSelect: ").strip()
    
    if choice == '1':
        idx_str = input("\nEnter case indices for video generation (e.g., 1, 3-5, all): ").strip().lower()
        if idx_str == 'all':
            indices = list(range(len(cases)))
        else:
            indices = parse_indices(idx_str, len(cases))
        
        if not indices:
            print("No valid indices selected.")
            return
        
        print(f"\nGenerating videos for {len(indices)} case(s)...")
        for i in indices:
            generate_video(cases[i])
    elif choice == '2':
        idx_str = input("\nEnter case indices for interface extraction (e.g., 1, 3-5, all): ").strip().lower()
        if idx_str == 'all':
            indices = list(range(len(cases)))
        else:
            indices = parse_indices(idx_str, len(cases))
        
        if not indices:
            print("No valid indices selected.")
            return
        
        print(f"\nExtracting interfaces for {len(indices)} case(s)...")
        for i in indices:
            extract_interface(cases[i])

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
    main_menu()
