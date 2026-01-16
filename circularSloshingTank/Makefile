# ====================================================================================
# Circular Sloshing Tank Simulation Makefile
# ====================================================================================
#
# USAGE:
#   make [target] [OVERRIDE_VAR=value]
#
# TARGETS:
#   all       : Generate mesh, motion file, and setup case (default).
#   mesh      : Generate geometry (.geo) and mesh (.msh) using Gmsh.
#   motion    : Generate 6DoF motion file (constant/6DoF.dat).
#   case      : Initialize field variables (runs update_setFields.py).
#   run       : Run the OpenFOAM simulation (gmshToFoam -> setFields -> foamRun).
#   clean     : Remove generated mesh, motion, and result files.
#   help      : Show this help message.
#
# PARAMETERS:
#   H             : Tank height (default: 1.0)
#   D             : Tank diameter (default: 0.5)
#   MESH_SIZE     : Mesh characteristic length (default: 0.05)
#   GEO_TYPE      : Geometry type: 'flat' or 'cap' (default: flat)
#   MOTION_R      : Motion amplitude radius (default: 0.1)
#   MOTION_FREQ   : Motion frequency in Hz (default: 0.5)
#   DURATION      : Simulation duration in seconds (default: 10.0)
#   DT            : Time step for motion file (default: 0.01)
#   RAMP_DURATION : Smooth start ramp duration (default: -1, i.e., 10% of total)
#
# ====================================================================================

# Parameters
H ?= 1.0
D ?= 0.5
MESH_SIZE ?= 0.05
MOTION_R ?= 0.1
MOTION_FREQ ?= 0.5
DURATION ?= 10.0
DT ?= 0.01
RAMP_DURATION ?= -1
GEO_TYPE ?= flat

.PHONY: all mesh motion case run clean help

all: prep run

help:
	@grep -E '^[a-zA-Z_-]+:.*?$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Read the header in Makefile for variable documentation."

# Prepare geometry and coordinate files (Run on Host/Login Node)
prep: motion case
	python3 generate_mesh.py $(H) $(D) $(MESH_SIZE) $(GEO_TYPE)
	gmsh -3 cylinder.geo -format msh2 -o cylinder.msh

# Generate Motion
motion:
	python3 generate_motion.py $(MOTION_R) $(MOTION_FREQ) $(DURATION) $(DT) $(RAMP_DURATION)

# Configure Case (Initial Fields)
case:
	# Update setFieldsDict for half-fill height (H/2)
	python3 update_setFields.py $(H)

# Run Simulation
OF_PREFIX ?= 
run:
	# commands to be run in OpenFOAM environment
	$(OF_PREFIX) gmshToFoam cylinder.msh
	$(OF_PREFIX) setFields
	$(OF_PREFIX) foamRun

clean:
	rm -f cylinder.geo cylinder.msh
	rm -f constant/6DoF.dat
	rm -rf 0/alpha.water.orig  # created by setFields usually backup
	# standard cleanup
