# Agent Knowledge Base: Circular Sloshing Tank

> [!IMPORTANT]
> **AGENT PROTOCOL**: All agents MUST consult this file before running simulations. If you solve a new bug or learn a technical quirk, you are REQUIRE to update this file with a new entry in the "Technical Lessons Learned" section.

This file tracks technical discoveries, version-specific quirks, and project-wide context for the OpenFOAM sloshing simulation.

## 📝 Technical Lessons Learned

### 1. OpenFOAM 13 (Foundation) - `setFieldsDict`
As of OpenFOAM 13, the `setFieldsDict` syntax has transitioned from list-based to dictionary-based. 

**Old Syntax (Deprecated/Unsupported):**
```cpp
defaultFieldValues ( volScalarFieldValue alpha.water 0 );
regions ( boxToCell { box (...); fieldValues ( volScalarFieldValue alpha.water 1 ); } );
```

**New Syntax (OpenFOAM 13):**
```cpp
defaultValues { alpha.water 0; }
zones { water { type box; box (...); values { alpha.water 1; } } }
```
*   `defaultFieldValues` -> `defaultValues`
*   `regions` -> `zones`
*   Selection (e.g., `boxToCell`) is replaced by a named selection block with a `type`.
*   `fieldValues` -> `values`

### 3. Soft Start Motion
To avoid infinite acceleration (velocity impulse) at $t=0$ caused by instantly applying the full motion amplitude (e.g., $x=r$ at $t=0$), we implemented a "Soft Start" mechanism.
*   **Mechanism**: A `smootherstep` function ($6\tau^5 - 15\tau^4 + 10\tau^3$) ramps the motion amplitude from 0 to $r$ over a defined `RAMP_DURATION` (defaults to 10% of simulation time).
*   **Result**: The simulation now starts with zero position, zero velocity, and zero acceleration, ensuring smooth transient behavior and better numerical stability.

### 2. Python-to-OpenFOAM Dictionary Generation
When generating OpenFOAM dictionaries using Python `f-strings`, curly braces `{}` must be escaped (doubled) as `{{}}` to prevent Python from interpreting them as variable placeholders.

## 🏃 Current Workflow

**Interactive Manager** (`python3 sloshing_manager.py`):

1.  **Build Case Setups**: 
    - Configure parameters (H, D, mesh, geo, R, freq, duration, dt, ramp)
    - Support for parameter sweeps (MATLAB-style ranges: `0.1:0.05:0.2`)
    - Generates `case_*` folders with mesh, motion, and initial conditions
    
2.  **Run Cases**:
    - Lists available cases with completion status
    - On Oscar: Submits to Slurm with dynamic resource allocation
    - Locally: Runs sequentially if OpenFOAM is installed
    
3.  **Postprocess**:
    - Generate MP4 videos (`pvpython` required)
    - Extract interface data (VTP files + CSV summary)

**Default Parameters**:
- Tank: H=0.1m, D=0.02m
- Mesh: 0.002m characteristic length
- Motion: 2.0Hz oscillation at 0.003m amplitude
- Duration: 10s with 10% soft-start ramp

## 🎯 Project Goals

## ✅ Simulation Verification Checklist
How to assess if the run was successful:

1.  **Courant Number (Co)**: Check the log for `max Courant Number`.
    -   **Good**: < 0.5 (Ideal for accuracy).
    -   **Acceptable**: < 0.9 (Standard stability limit).
    -   **Bad**: > 1.0 (Likely to crash or produce non-physical results).
2.  **Phase Boundedness**:
    -   `alpha.water` should never exceed 1.0 or drop below 0.0.
    -   Watch for "bounding alpha.water" warnings in the log.
3.  **Mass Conservation**:
    -   The total volume of water should not change significantly.
    -   Check `time step continuity errors` in the log.
4.  **Visual Inspection (ParaView)**:
    -   Is the interface sharp? (Smearing indicates excessive diffusion).
    -   Does the water surface move smoothly? (Jaggedness suggests mesh issues).
