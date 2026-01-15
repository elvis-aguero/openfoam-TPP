# ====================================================================================
# OpenFOAM Case Generator & Runner
# ====================================================================================
#
# USAGE:
#   make [target] [OVERRIDE_VAR=value]
#
# DESCRIPTION:
#   This Makefile automatically generates a new case folder based on your parameters.
#   It copies the 'circularSloshingTank' template, configures it, and runs the simulation.
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
# EXAMPLE:
#   make run H=2.0 GEO_TYPE=cap
#   -> Creates folder: case_H2.0_D0.5_cap_R0.1_f0.5
#   -> Runs simulation inside that folder
#
# ====================================================================================

# Defaults
H ?= 1.0
D ?= 0.5
MESH_SIZE ?= 0.05
MOTION_R ?= 0.1
MOTION_FREQ ?= 0.5
DURATION ?= 10.0
DT ?= 0.01
RAMP_DURATION ?= -1
GEO_TYPE ?= flat
TEMPLATE_DIR := circularSloshingTank

# Generate unique case name
CASE_NAME := case_H$(H)_D$(D)_$(GEO_TYPE)_R$(MOTION_R)_f$(MOTION_FREQ)

.PHONY: all help run clean

all: run

help:
	@echo "OpenFOAM Case Generator"
	@echo "-----------------------"
	@echo "Current Configuration:"
	@echo "  CASE_NAME   : $(CASE_NAME)"
	@echo "  TEMPLATE    : $(TEMPLATE_DIR)"
	@echo "  H (Height)  : $(H)"
	@echo "  D (Diam)    : $(D)"
	@echo "  GEO_TYPE    : $(GEO_TYPE)"
	@echo ""
	@echo "Usage:"
	@echo "  make run        : Generate case folder and run simulation."
	@echo "  make clean      : Remove ALL generated case_* folders."

# Check if OpenFOAM environment is present
HAS_OPENFOAM := $(shell command -v gmshToFoam 2> /dev/null)

run:
	@echo "============================================================"
	@echo " Generating Case: $(CASE_NAME)"
	@echo "============================================================"
	@if [ -d "$(CASE_NAME)" ]; then \
		echo "Directory $(CASE_NAME) already exists. Skipping copy."; \
	else \
		echo "Copying template to $(CASE_NAME)..."; \
		cp -r $(TEMPLATE_DIR) $(CASE_NAME); \
	fi
	@if [ -z "$(HAS_OPENFOAM)" ]; then \
		echo "⚠️  OpenFOAM commands (gmshToFoam) not found."; \
		echo "✅  Case folder '$(CASE_NAME)' created successfully."; \
		echo "    (Skipping simulation run)"; \
	else \
		echo "Entering $(CASE_NAME) and running simulation..."; \
		$(MAKE) -C $(CASE_NAME) run \
			H=$(H) \
			D=$(D) \
			MESH_SIZE=$(MESH_SIZE) \
			GEO_TYPE=$(GEO_TYPE) \
			MOTION_R=$(MOTION_R) \
			MOTION_FREQ=$(MOTION_FREQ) \
			DURATION=$(DURATION) \
			DT=$(DT) \
			RAMP_DURATION=$(RAMP_DURATION); \
	fi

clean:
	@echo "Removing all generated case folders..."
	rm -rf case_*
