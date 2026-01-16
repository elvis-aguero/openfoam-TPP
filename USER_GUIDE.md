# 🌊 Circular Sloshing Tank - User Guide

Welcome! This guide will help you run the sloshing simulation without needing to be an OpenFOAM expert.

## 🚀 Quick Start
To run the standard simulation (Flat bottom tank, default settings), open your terminal in the **root folder** (where `Makefile` is) and type:

```bash
make run
```
This will:
1.  **Create a new folder** (e.g., `case_H1.0_D0.5_flat_...`).
2.  Copy the template files into it.
3.  Generate the mesh and motion files inside that folder.
4.  Run the simulation.

---

## 🧪 Common "Recipes"

Here are the commands for specific scenarios you might want to test.

### 1. "I want the Round Bottom (Hemispherical) Tank"
If you want to simulate the tank with a round cap at the bottom instead of a flat one:

```bash
make run GEO_TYPE=cap
```

### 2. "I want to change the Tank Height/Depth"
To change the tank height (`H`) or diameter (`D`):
*   `H`: Height of the tank (default 1.0 meter).
*   `D`: Diameter of the tank (default 0.5 meters).

Example: A taller, narrower tank:
```bash
make run H=2.0 D=0.3
```

### 3. "I want to change the Motion"
You can adjust how much it shakes (`MOTION_R`) and how fast (`MOTION_FREQ`):
*   `MOTION_R`: Amplitude of the shake (in meters).
*   `MOTION_FREQ`: Frequency of the shake (in Hz).

Example: Faster, smaller shake:
```bash
make run MOTION_FREQ=1.0 MOTION_R=0.05
```

### 4. "I want a Longer Simulation"
By default, it runs for 10 seconds.
*   `DURATION`: Total time in seconds.

Example: Run for 30 seconds:
```bash
make run DURATION=30.0
```

### 5. "I want to adjust the Soft Start (Ramp)"
To avoid "impulsive" starts, the simulation ramps up the motion amplitude.
*   `RAMP_DURATION`: Time in seconds to reach full amplitude. 
*   **Default**: 10% of the total duration (e.g., 1.0s ramp for a 10s run).

Example: A very slow 5-second ramp:
```bash
make run RAMP_DURATION=5.0
```

---

## 📊 How to View Results

1.  Open **ParaView**.
2.  Open the file named `case.foam` located in this folder.
3.  In ParaView:
    *   Click **Apply** (green button).
    *   In the "Fields" list, check `alpha.water`.
    *   Press the **Play** button (▶️) to watch the animation.

---

## 🧹 Cleaning Up
If you want to delete all previous results and start fresh:

```bash
make clean
```
(*It is a good practice to run this before starting a completely different configuration.*)

---

## 💻 Running Locally vs on Oscar
The `Makefile` is smart enough to detect where you are:
*   **On your personal Mac/Linux**: If you have OpenFOAM installed, `make run` will build the folder **and** start the simulation.
*   **On the Oscar Login Node**: If you run `make run`, it will safely **build the folders** but skip the simulation (to avoid using CPU on the login node). You can then use the Slurm Manager to submit them to compute nodes.

---

## 🏛️ HPC / Slurm Management (Oscar @ CCV)

If you are running on the **Oscar cluster** (Brown University), you can use the interactive Slurm manager to queue your simulations.

1.  **Generate your cases** first using the `make` commands listed above.
2.  **Run the manager**:
    ```bash
    python3 manage_slurm.py
    ```
3.  **Choose your mode**:
    *   **Option 1 (Single)**: Browse and select a specific case to submit.
    *   **Option 2 (Run All)**: Automatically queue every `case_*` folder in the directory.

**Note**: The manager is configured to use your `of13` Apptainer wrapper automatically. Each job defaults to 1 core, **64GB of RAM**, and **72 hours**. You can adjust these settings by editing the `SLURM_DEFAULTS` at the top of `manage_slurm.py`.
