#!/usr/bin/env python3
import os
import sys
import subprocess

# --- Configuration for Oscar (CCV Brown University) ---
# Base defaults if estimation fails
SLURM_DEFAULTS = {
    "partition": "batch",
    "nodes": 1,
    "cpus": 1,
    "mem": "8G", 
    "time": "24:00:00",
}

# The user uses 'of13' wrapper to execute OpenFOAM commands via Apptainer
OF_EXEC = "of13"

def parse_case_params(case_name):
    """
    Extracts H and D from case name to estimate mesh size.
    Format: case_H{H}_D{D}_...
    """
    try:
        parts = case_name.split('_')
        h = float(parts[1].replace('H', ''))
        d = float(parts[2].replace('D', ''))
        return h, d
    except (IndexError, ValueError):
        return 0.1, 0.02 # Fallback to defaults

def estimate_resources(case_dir):
    """
    Estimates required memory and time based on domain volume.
    Assumes a fixed mesh characteristic length (default 0.002m from Makefile).
    """
    h, d = parse_case_params(os.path.basename(case_dir))
    
    # Calculate Volume (Cylinder)
    r = d / 2.0
    vol = 3.14159 * (r**2) * h
    
    # Estimate Cell Count
    # Mesh size is now 0.002m (2mm). Cell volume approx (mesh_size)^3
    # This is a rough heuristic.
    mesh_size = 0.002
    cell_vol = mesh_size ** 3
    n_cells = vol / cell_vol
    
    # Heuristics for OpenFOAM:
    # ~1GB RAM per 1M cells is conservative.
    # Time scales with N_cells and Duration. 
    
    # Memory
    # Base 4GB + 1GB per 500k cells
    req_mem_gb = 4 + (n_cells / 500000)
    
    # Time
    # Base 4 hours + scaling
    # We'll validly cap these since sloshing is expensive.
    
    if n_cells < 100000: # Small case
        mem = "8G"
        time = "24:00:00"
    elif n_cells < 1000000: # Medium
        mem = "32G"
        time = "48:00:00"
    else: # Large
        mem = "64G"
        time = "72:00:00"
        
    print(f"   📊 Est. Cells: {int(n_cells):,} | Allocating: {mem} RAM, {time}")
    
    return {"mem": mem, "time": time}

def find_cases():
    """Scans the current directory for folders starting with 'case_'"""
    cases = [d for d in os.listdir('.') if os.path.isdir(d) and d.startswith('case_')]
    return sorted(cases)

def create_slurm_script(case_dir):
    """Creates a .slurm submission script inside the case directory"""
    script_path = os.path.join(case_dir, "run_simulation.slurm")
    
    # Estimate resources dynamically
    resources = estimate_resources(case_dir)
    
    # Construct the SBATCH header as per user's template
    header = [
        "#!/usr/bin/env bash",
        f"#SBATCH -J {case_dir}",
        f"#SBATCH -p {SLURM_DEFAULTS['partition']}",
        f"#SBATCH -N {SLURM_DEFAULTS['nodes']}",
        f"#SBATCH -n {SLURM_DEFAULTS['cpus']}",
        f"#SBATCH --time={resources['time']}",
        f"#SBATCH --mem={resources['mem']}",
        f"#SBATCH -o {case_dir}/slurm.%j.out",
        f"#SBATCH -e {case_dir}/slurm.%j.err",
        "",
        "set -euo pipefail",
        "export OMP_NUM_THREADS=1",
        "",
        "echo \"Start: $(date)\"",
        "echo \"Running in: $(pwd)\"",
        f"echo \"Case:  {case_dir}\"",
        "",
        f"# Call the simulation with the of13 prefix",
        f"make -C {case_dir} run OSCAR=1",
        "",
        "echo \"End:   $(date)\""
    ]
    
    with open(script_path, 'w') as f:
        f.write("\n".join(header))
    
    return script_path

def submit_job(case_dir):
    """Generates and submits the slurm job"""
    script_path = create_slurm_script(case_dir)
    try:
        # We run the command from the root to ensure paths match $SLURM_SUBMIT_DIR
        result = subprocess.run(['sbatch', script_path], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Submitted {case_dir}: {result.stdout.strip()}")
        else:
            print(f"❌ Failed to submit {case_dir}: {result.stderr.strip()}")
    except FileNotFoundError:
        print(f"⚠️  'sbatch' command not found. (Not on a Slurm system?)")
        print(f"   Script created: {script_path}")

def main():
    cases = find_cases()
    
    if not cases:
        print("No simulation folders found (expected 'case_*').")
        print("Please generate some cases first using 'make run'.")
        return

    print("\n" + "="*40)
    print("   Oscar Slurm Job Manager")
    print("="*40)
    print("1) Single Simulation Mode")
    print("2) Run All Mode (Batch Queue)")
    print("Q) Quit")
    
    choice = input("\nSelect an option: ").strip().lower()

    if choice == '1':
        print("\nAvailable Simulations:")
        for idx, case in enumerate(cases):
            print(f"  {idx + 1}) {case}")
        
        try:
            val = input("\nEnter simulation number (or name): ").strip()
            if val.isdigit():
                selected_case = cases[int(val) - 1]
            else:
                selected_case = val if val in cases else None
            
            if selected_case:
                submit_job(selected_case)
            else:
                print("Invalid selection.")
        except Exception as e:
            print(f"Error: {e}")

    elif choice == '2':
        confirm = input(f"Submit all {len(cases)} simulations to the queue? [y/N]: ").strip().lower()
        if confirm == 'y':
            for case in cases:
                submit_job(case)
        else:
            print("Batch submission cancelled.")
            
    elif choice == 'q':
        print("Exiting.")
    else:
        print("Invalid option.")

if __name__ == "__main__":
    main()
