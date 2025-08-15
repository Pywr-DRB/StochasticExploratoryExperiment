#!/bin/bash
#SBATCH --job-name=Figs
#SBATCH --output=./logs/fig_generation.out
#SBATCH --error=./logs/fig_generation.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
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

# Workflow control flags 
PLOT_DIAGNOSTICS=${PLOT_DIAGNOSTICS:-false}
PLOT_OUTCOMES=${PLOT_OUTCOMES:-true}

# make directories
mkdir -p logs figures

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
echo "========================================================================"


# Step 5: Generate ensemble diagnostics (single core)
if [ "$PLOT_DIAGNOSTICS" = true ]; then
    echo "STEP 5: Generating ensemble diagnostics..."
    echo "Starting at: $(date)"
    
    python3 02_plot_ensemble_diagnostics.py
    
    if [ $? -eq 0 ]; then
        echo "✓ Ensemble diagnostics completed successfully"
    else
        echo "✗ Ensemble diagnostics failed"
        exit 1
    fi
    
    echo "Completed at: $(date)"
    echo "----------------------------------------"
fi

# Step 6: Analyze outcomes and generate plots (single core)
if [ "$PLOT_OUTCOMES" = true ]; then
    echo "STEP 6: Analyzing outcomes and generating plots..."
    echo "Starting at: $(date)"
    
    echo "  Calculating Hashimoto metrics..."
    # python3 999_calculate_hashimoto_metrics.py
    
    echo "Plotting pywrdrb dynamics..."
    python3 999_plot_pywrdrb_dynamics.py

    echo "Completed at: $(date)"
    echo "----------------------------------------"
fi

# =============================================================================
# WORKFLOW COMPLETION
# =============================================================================

echo "========================================================================"
echo "STATIONARY STOCHASTIC ENSEMBLE WORKFLOW COMPLETED"
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
    spec = get_ensemble_set_spec(i)
    gage_exists = '✓' if os.path.exists(spec.files['gage_flow']) else '✗'
    inflow_exists = '✓' if os.path.exists(spec.files['catchment_inflow']) else '✗'
    output_exists = '✓' if os.path.exists(spec.output_file) else '✗'
    print(f'  Set {i+1}: Gage {gage_exists} | Inflow {inflow_exists} | Output {output_exists}')

"


echo "========================================================================"
echo "WORKFLOW COMPLETE - CHECK LOGS FOR DETAILED RESULTS"
echo "========================================================================"