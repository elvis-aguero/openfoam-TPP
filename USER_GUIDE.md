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
