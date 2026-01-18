# OpenFOAM Sloshing Simulation Manager

Interactive Python tool for managing orbital shaking simulations in cylindrical containers using OpenFOAM.

## Quick Start

```bash
python3 main.py
```

The script will automatically:
1. Create a virtual environment (`sloshing/`) if needed
2. Install dependencies (numpy, scipy)
3. Launch the interactive menu

## Features

- **Build Cases**: Parameter sweeps with MATLAB-style ranges
- **Run Cases**: Local execution or Slurm submission (Oscar)
- **Postprocess**: 
  - Generate videos (requires ParaView)
  - Extract interface data (VTP + CSV)
  - Potential flow theory predictions

## Requirements

- Python 3.8+
- OpenFOAM 13 (for running simulations)
- Gmsh (for mesh generation)
- ParaView (optional, for postprocessing)

Dependencies are auto-installed on first run.

## Documentation

- `USER_GUIDE.md` - Usage instructions
- `circularSloshingTank/KNOWLEDGE_BASE.md` - Technical notes
