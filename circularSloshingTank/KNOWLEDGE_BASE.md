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
1.  **Mesh (`make mesh`)**: Python script generates a `.geo` file for Gmsh, which is then meshed and converted to OpenFOAM format.
2.  **Motion (`make motion`)**: Prescribes oscillation frequencies and amplitudes into `constant/6DoF.dat`.
3.  **Setup (`make case`)**: Runs `update_setFields.py` to match the liquid level to the tank height (default H/2).
4.  **Initialize (`setFields`)**: MUST be run before solving to populate `alpha.water`.
5.  **Solve (`foamRun`)**: Runs the `incompressibleVoF` solver.

## 🎯 Project Goals
- **Objective**: Study liquid resonance in circular geometry.
- **Parameters**: 
  - Tank: H=1.0, D=0.5.
  - Mesh: Standard refinement at 0.05.
  - Motion: 0.5Hz oscillation at 0.1m amplitude.
- **Outcome**: Capture the free surface deformation and damping coefficients.

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
