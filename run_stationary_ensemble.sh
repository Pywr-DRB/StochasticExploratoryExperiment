#!/bin/bash
#SBATCH --job-name=SE-DRB
#SBATCH --output=./logs/exploratory.out
#SBATCH --error=./logs/exploratory.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=40
#SBATCH --time=48:00:00
#SBATCH --mem=0

# Hierarchical ensemble workflow for large-scale stochastic simulation
# Processes ensemble sets independently with full MPI parallelization

# =============================================================================
# SETUP AND CONFIGURATION
# =============================================================================

# Load modules and environment
module load python/3.11.5
source venv/bin/activate
np=$(($SLURM_NTASKS_PER_NODE * $SLURM_NNODES))

# Read configuration from Python config file
eval $(python3 -c "
from config import *
print(f'TOTAL_REALIZATIONS={TOTAL_REALIZATIONS}')
print(f'N_ENSEMBLE_SETS={N_ENSEMBLE_SETS}')
print(f'N_REALIZATIONS_PER_ENSEMBLE_SET={N_REALIZATIONS_PER_ENSEMBLE_SET}')
print(f'N_REALIZATIONS_PER_PYWRDRB_BATCH={N_REALIZATIONS_PER_PYWRDRB_BATCH}')
")


# Workflow control flags (can be overridden via environment variables)
RUN_BASELINE=${RUN_BASELINE:-false}
GENERATE_ENSEMBLE_SETS=${GENERATE_ENSEMBLE_SETS:-true}
PLOT_DIAGNOSTICS=${PLOT_DIAGNOSTICS:-false}
PREP_PYWRDRB=${PREP_PYWRDRB:-true}
RUN_PYWRDRB=${RUN_PYWRDRB:-true}
PLOT_OUTCOMES=${PLOT_OUTCOMES:-true}

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
echo "PyWR-DRB batch size: $N_REALIZATIONS_PER_PYWRDRB_BATCH"
echo "========================================================================"

# Print Python configuration summary
python3 -c "from config import print_experiment_summary; print_experiment_summary()"

echo "========================================================================"

# Step 1: Run baseline simulations (if requested)
if [ "$RUN_BASELINE" = true ]; then
    echo "STEP 1: Running baseline simulations..."
    echo "Starting at: $(date)"
    
    time mpirun -np $np python3 00_run_baseline_simulations.py
    
    if [ $? -eq 0 ]; then
        echo "✓ Baseline simulations completed successfully"
    else
        echo "✗ Baseline simulations failed"
        exit 1
    fi
    
    echo "Completed at: $(date)"
    echo "----------------------------------------"
fi

# Step 2: Generate all ensemble sets in parallel
if [ "$GENERATE_ENSEMBLE_SETS" = true ]; then
    echo "STEP 2: Generating ensemble sets in parallel..."
    echo "Starting at: $(date)"
    
    time mpirun -np $np python3 01_generate_stationary_ensemble_sets.py
    
    if [ $? -eq 0 ]; then
        echo "✓ Ensemble generation completed successfully"
    else
        echo "✗ Ensemble generation failed"
        exit 1
    fi
    
    echo "Completed at: $(date)"
    echo "----------------------------------------"
fi

# Step 3: Prepare PyWR-DRB inputs for all sets in parallel
if [ "$PREP_PYWRDRB" = true ]; then
    echo "STEP 3: Preparing PyWR-DRB inputs for all ensemble sets..."
    echo "Starting at: $(date)"
    
    time mpirun -np $np python3 03_prep_pywrdrb_inputs.py
    
    if [ $? -eq 0 ]; then
        echo "✓ PyWR-DRB input preparation completed successfully"
    else
        echo "✗ PyWR-DRB input preparation failed"
        exit 1
    fi
    
    echo "Completed at: $(date)"
    echo "----------------------------------------"
fi

# Step 4: Run PyWR-DRB simulations for all sets in parallel
if [ "$RUN_PYWRDRB" = true ]; then
    echo "STEP 4: Running PyWR-DRB simulations for all ensemble sets..."
    echo "Starting at: $(date)"
    
    time mpirun -np $np python3 03_run_pywrdrb_simulations.py
    
    if [ $? -eq 0 ]; then
        echo "✓ PyWR-DRB simulations completed successfully"
    else
        echo "✗ PyWR-DRB simulations failed"
        exit 1
    fi
    
    echo "Completed at: $(date)"
    echo "----------------------------------------"
fi

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
    python3 999_calculate_hashimoto_metrics.py
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Hashimoto metrics calculated"
    else
        echo "  ✗ Hashimoto metrics calculation failed"
    fi
    
    echo "  Plotting drought violations..."
    python3 999_plot_drought_montague_violations.py
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Drought violation plots generated"
    else
        echo "  ✗ Drought violation plotting failed"
    fi
    
    echo "  Plotting PyWR-DRB dynamics..."
    python3 999_plot_pywrdrb_dynamics.py
    
    if [ $? -eq 0 ]; then
        echo "  ✓ PyWR-DRB dynamics plots generated"
    else
        echo "  ✗ PyWR-DRB dynamics plotting failed"
    fi
    
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

print('')
print('Other output files:')
baseline_exists = '✓' if os.path.exists(RECONSTRUCTION_OUTPUT_FNAME) else '✗'
print(f'  Baseline: {baseline_exists}')
"


echo "========================================================================"
echo "WORKFLOW COMPLETE - CHECK LOGS FOR DETAILED RESULTS"
echo "========================================================================"