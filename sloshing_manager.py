#!/usr/bin/env python3
import os
import sys
import shutil
import subprocess
import argparse
import math

# --- Constants & Defaults ---
TEMPLATE_DIR = "circularSloshingTank"
MESH_SIZE_DEFAULT = 0.002
DURATION_DEFAULT = 10.0
DT_DEFAULT = 0.001

def get_env_info():
    """Detects the environment: Oscar, VM (Native OF), or Bare."""
    has_sbatch = shutil.which("sbatch") is not None
    has_of13 = shutil.which("of13") is not None
    has_foam_run = shutil.which("foamRun") is not None
    
    if has_sbatch and has_of13:
        return "oscar"
    elif has_foam_run:
        return "vm"
    else:
        return "bare"

def estimate_resources(h, d, mesh_size):
    """Estimates required memory and time based on domain volume."""
    vol = math.pi * ((d / 2.0)**2) * h
    cell_vol = mesh_size ** 3
    n_cells = vol / cell_vol
    
    if n_cells < 100000: # Small
        return "8G", "24:00:00", n_cells
    elif n_cells < 1000000: # Medium
        return "32G", "48:00:00", n_cells
    else: # Large
        return "64G", "72:00:00", n_cells

def setup_case(args):
    """Creates the case directory and runs setup scripts."""
    case_name = f"case_H{args.H}_D{args.D}_{args.geo}_R{args.R}_f{args.freq}"
    
    # 1. Copy Template
    if os.path.exists(case_name):
        print(f"⚠️ Case {case_name} already exists. Skipping copy.")
    else:
        print(f"📂 Creating case: {case_name}")
        shutil.copytree(TEMPLATE_DIR, case_name)
        # Ensure writable
        for root, dirs, files in os.walk(case_name):
            for d in dirs:
                os.chmod(os.path.join(root, d), 0o777)
            for f in files:
                os.chmod(os.path.join(root, f), 0o666)

    # 2. Run Setup Scripts (Local Python)
    print("🔧 Running setup scripts...")
    cwd = os.path.join(os.getcwd(), case_name)
    
    # Motion
    subprocess.run([
        sys.executable, "generate_motion.py", 
        str(args.R), str(args.freq), str(args.duration), str(args.dt), str(args.ramp)
    ], cwd=cwd, check=True)
    
    # Fields (setFieldsDict update)
    subprocess.run([sys.executable, "update_setFields.py", str(args.H)], cwd=cwd, check=True)
    
    # Mesh (.geo to .msh)
    subprocess.run([
        sys.executable, "generate_mesh.py", 
        str(args.H), str(args.D), str(args.mesh), args.geo
    ], cwd=cwd, check=True)
    
    # 3. Run Gmsh (Local or VM)
    gmsh_path = shutil.which("gmsh")
    if gmsh_path:
        print("🕸️ Generating mesh with Gmsh...")
        subprocess.run([
            "gmsh", "-3", "cylinder.geo", "-format", "msh2", "-o", "cylinder.msh"
        ], cwd=cwd, check=True)
    else:
        print("⚠️ gmsh not found in PATH. Skipping .msh generation.")

    return case_name

def run_oscar(case_name, args):
    """Submits job to Slurm on Oscar."""
    mem, time_limit, n_cells = estimate_resources(args.H, args.D, args.mesh)
    script_path = os.path.join(case_name, "run_simulation.slurm")
    
    header = [
        "#!/usr/bin/env bash",
        f"#SBATCH -J {case_name}",
        f"#SBATCH -p batch",
        f"#SBATCH -N 1",
        f"#SBATCH -n 1",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH -o {case_name}/slurm.%j.out",
        f"#SBATCH -e {case_name}/slurm.%j.err",
        "",
        "set -euo pipefail",
        "export OMP_NUM_THREADS=1",
        "",
        "echo \"Start: $(date)\"",
        f"echo \"Case:  {case_name}\"",
        "",
        f"make -C {case_name} run OSCAR=1",
        "",
        "echo \"End:   $(date)\""
    ]
    
    with open(script_path, "w") as f:
        f.write("\n".join(header))
    
    print(f"📊 Est. Cells: {int(n_cells):,}")
    print(f"🚀 Submitting to Slurm ({mem}, {time_limit})...")
    subprocess.run(["sbatch", script_path], check=True)

def run_local(case_name):
    """Runs simulation locally in VM."""
    print(f"🏃 Starting simulation in {case_name}...")
    subprocess.run(["make", "-C", case_name, "run"], check=True)

def main():
    parser = argparse.ArgumentParser(description="Unified Sloshing Case Manager")
    parser.add_argument("--H", type=float, default=0.1, help="Tank height (m)")
    parser.add_argument("--D", type=float, default=0.02, help="Tank diameter (m)")
    parser.add_argument("--mesh", type=float, default=MESH_SIZE_DEFAULT, help="Mesh size")
    parser.add_argument("--geo", choices=["flat", "cap"], default="flat", help="Geometry type")
    parser.add_argument("--R", type=float, default=0.003, help="Motion radius (m)")
    parser.add_argument("--freq", type=float, default=2.0, help="Motion frequency (Hz)")
    parser.add_argument("--duration", type=float, default=DURATION_DEFAULT, help="Duration (s)")
    parser.add_argument("--dt", type=float, default=DT_DEFAULT, help="Motion DT (s)")
    parser.add_argument("--ramp", type=float, default=-1, help="Ramp duration (s, -1 for 10%%)")
    parser.add_argument("--setup-only", action="store_true", help="Only build the case, don't run")
    parser.add_argument("--run", action="store_true", help="Run after setup (detects environment)")
    parser.add_argument("--all", action="store_true", help="Queue all existing cases (Oscar only)")

    args = parser.parse_args()
    env = get_env_info()
    
    if args.all:
        if env != "oscar":
            print("❌ --all mode is only for Oscar.")
            return
        cases = [d for d in os.listdir('.') if os.path.isdir(d) and d.startswith('case_')]
        for case in cases:
            # We need to re-parse params to get resources right, or just use defaults
            # For simplicity, we just submit them
            run_oscar(case, args)
        return

    # Standard setup & run
    case_name = setup_case(args)
    
    if args.run:
        if env == "oscar":
            run_oscar(case_name, args)
        elif env == "vm":
            run_local(case_name)
        else:
            print("❌ OpenFOAM not detected. Setup complete, but cannot run simulation.")

if __name__ == "__main__":
    main()
