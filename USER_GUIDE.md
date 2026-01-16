
Welcome! This guide will help you run the sloshing simulation using the **Sloshing Manager**, a unified tool for case setup and execution.

## 🚀 Quick Start
To run the standard simulation (Default settings), open your terminal in the **root folder** and type:

```bash
python3 sloshing_manager.py --run
```

This will:
1.  **Create a new case folder** (e.g., `case_H0.1_D0.02_...`).
2.  Run all setup scripts (Mesh, Motion, Initial Fields).
3.  **Automatically Run** the simulation (if on Oscar, it will queue a Slurm job; if in a VM, it will run locally).

---

## 🧪 Common "Recipes"

Use flags to override default values.

### 1. "I want to change the Tank Geometry"
```bash
# Spherical Cap bottom
python3 sloshing_manager.py --run --geo cap

# Flat bottom (default)
python3 sloshing_manager.py --run --geo flat
```

### 2. "I want to change the Dimensions"
*   `--H`: Height (m)
*   `--D`: Diameter (m)

Example:
```bash
python3 sloshing_manager.py --run --H 0.15 --D 0.03
```

### 3. "I want to change the Motion"
*   `--R`: Amplitude radius (m)
*   `--freq`: Frequency (Hz)

Example:
```bash
python3 sloshing_manager.py --run --freq 1.5 --R 0.005
```

### 4. "I want to change the Simulation Time"
*   `--duration`: Total time (s)
*   `--ramp`: Soft-start ramp duration (s)

Example:
```bash
python3 sloshing_manager.py --run --duration 30.0 --ramp 5.0
```

---

## 📊 How to View Results

1.  Open **ParaView**.
2.  Open the file named `case.foam` located inside your specific `case_...` folder.
3.  Click **Apply**, check `alpha.water`, and press **Play** (▶️).

---

## 🏛️ HPC / Slurm Management (Oscar @ CCV)

The manager is smart! When you use `--run` on Oscar:
1.  It **estimates resources** (RAM and Time) based on your mesh size.
2.  It creates a `.slurm` script **inside** the case folder.
3.  It **submits** the job automatically.

**Batch Mode**: To submit every case in the directory to the queue:
```bash
python3 sloshing_manager.py --all
```

---

## 🧹 Cleaning Up
To remove all generated cases and start fresh:
```bash
rm -rf case_*
```
