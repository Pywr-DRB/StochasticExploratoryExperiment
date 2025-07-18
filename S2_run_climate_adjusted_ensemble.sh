#!/bin/bash
#SBATCH --job-name=CAE
#SBATCH --output=./logs/climate_adjusted.out
#SBATCH --error=./logs/climate_adjusted.err
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=40
#SBATCH --time=48:00:00
#SBATCH --mem=0

# =============================================================================
# SETUP AND CONFIGURATION
# =============================================================================

# Load modules and environment
module load python/3.11.5
source venv/bin/activate
np=$(($SLURM_NTASKS_PER_NODE * $SLURM_NNODES))

# Read configuration from Python config file
eval $(python3 -c "from config import *")

# Workflow control flags (can be overridden via environment variables)
GENERATE_ENSEMBLE_SETS=${GENERATE_ENSEMBLE_SETS:-true}
PREP_PYWRDRB=${PREP_PYWRDRB:-true}
RUN_PYWRDRB=${RUN_PYWRDRB:-true}

# make directories
mkdir -p logs pywrdrb/inputs pywrdrb/outputs pywrdrb/models figures

# =============================================================================
# WORKFLOW
# =============================================================================

echo "========================================================================"
echo "STATIONARY STOCHASTIC ENSEMBLE WORKFLOW STARTING"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total MPI ranks: $np"
echo "Total realizations: $TOTAL_REALIZATIONS"
echo "Ensemble sets: $N_ENSEMBLE_SETS"
echo "Realizations per set: $N_REALIZATIONS_PER_ENSEMBLE_SET"
echo "Pywr-DRB batch size: $N_REALIZATIONS_PER_PYWRDRB_BATCH"
echo "========================================================================"

# Print Python configuration summary
python3 -c "from config import print_experiment_summary; print_experiment_summary('climate_adjusted')"

echo "========================================================================"

# Step 2: Generate all ensemble sets in parallel
if [ "$GENERATE_ENSEMBLE_SETS" = true ]; then
    echo "STEP 2: Generating ensemble sets in parallel..."
    echo "Starting at: $(date)"
    
    time mpirun -np $np python3 01_generate_stationary_ensemble_sets.py "climate_adjusted"
    
    echo "Completed at: $(date)"
    echo "----------------------------------------"
fi


### Prepare Pywr-DRB inputs for all sets in parallel
if [ "$PREP_PYWRDRB" = true ]; then
    echo "STEP 3: Preparing Pywr-DRB inputs for all ensemble sets..."
    echo "Starting at: $(date)"
    
    time mpirun -np $np python3 03_prep_pywrdrb_inputs.py "climate_adjusted"
    
    echo "Completed at: $(date)"
    echo "----------------------------------------"
fi

# Step 4: Run Pywr-DRB simulations for all sets in parallel
if [ "$RUN_PYWRDRB" = true ]; then
    echo "STEP 4: Running Pywr-DRB simulations for all ensemble sets..."
    echo "Starting at: $(date)"
    
    time mpirun -np $np python3 03_run_pywrdrb_simulations.py "climate_adjusted"

    echo "Completed at: $(date)"
    echo "----------------------------------------"
fi


# =============================================================================
# WORKFLOW COMPLETION
# =============================================================================

echo "========================================================================"
echo "CLIMATE-ADJUSTED ENSEMBLE WORKFLOW COMPLETED"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Completed at: $(date)"

# Print final summary
echo ""
echo "FINAL SUMMARY:"
echo "Total realizations generated: $TOTAL_REALIZATIONS"
echo "Ensemble sets: $N_ENSEMBLE_SETS"

# Check output files
echo ""
echo "OUTPUT FILES:"
python3 -c "
from config import *
import os
print('Ensemble set files:')
for i in range(N_ENSEMBLE_SETS):
    spec = get_ensemble_set_spec(i, 'climate_adjusted')
    gage_exists = 'SUCCESS' if os.path.exists(spec.files['gage_flow']) else 'FAIL'
    inflow_exists = 'SUCCESS' if os.path.exists(spec.files['catchment_inflow']) else 'FAIL'
    output_exists = 'SUCCESS' if os.path.exists(spec.output_file) else 'FAIL'
    print(f'  Set {i+1}: Gage {gage_exists} | Inflow {inflow_exists} | Output {output_exists}')
"

echo "========================================================================"
echo "WORKFLOW COMPLETE - CHECK LOGS FOR DETAILED RESULTS"
echo "========================================================================"