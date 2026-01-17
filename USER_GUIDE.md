# 🌊 Circular Sloshing Tank - User Guide

Welcome! This guide will help you run the sloshing simulation using the **interactive Sloshing Manager**.

## 🚀 Quick Start
Run the manager:
```bash
python3 sloshing_manager.py
```

You will be prompted:
1.  **Are you on Oscar?** (Y/N) - This determines whether jobs are submitted to Slurm.
2.  **Main Menu**:
    *   **1) Build Case Setups**: Create one or more simulation cases.
    *   **2) Run Cases**: Run or submit selected cases.
    *   **3) Postprocess**: (Coming soon).

---

## 🛠️ Building Cases (Sweeps)

When you select "Build Case Setups", you can:
1.  **Use defaults**: Just press Enter at the prompt to generate one default case.
2.  **Override a single parameter**: Enter the parameter name (e.g., `H`) and a new value.
3.  **Sweep over a parameter**: Enter a list like `0.1, 0.15, 0.2` or a MATLAB-style range like `0.1:0.05:0.2`.

**Zip vs Cartesian**:
*   If all sweep lists have the **same length**, they are **zipped** (paired 1-to-1).
*   If lengths **differ**, a **Cartesian product** is generated (all combinations). You will be asked to confirm.

---

## 🏃 Running Cases

When you select "Run Cases":
1.  The manager scans for `case_*` folders.
2.  It shows the status: cases that are complete will have `(DONE)` next to them.
3.  Enter the indices of cases you want to run (e.g., `1, 3-5, all`).
4.  On **Oscar**: Jobs are submitted to Slurm with smart resource allocation.
5.  **Locally**: If OpenFOAM is installed, simulations run sequentially.

---

## 📊 Viewing Results (ParaView)

1.  Open **ParaView**.
2.  Open `case.foam` inside your `case_...` folder.
3.  Click **Apply**, check `alpha.water`, and press **Play** (▶️).

---

## 🧹 Cleaning Up
```bash
rm -rf case_*
```
