#!/bin/bash
#SBATCH --job-name=CAE
#SBATCH --output=./logs/CAE.out
#SBATCH --error=./logs/CAE.err
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=30
#SBATCH --time=144:00:00
#SBATCH --mem=0
#SBATCH --exclusive

# Setup
module load python/3.11.5
source venv/bin/activate
np=$(($SLURM_NTASKS_PER_NODE * $SLURM_NNODES))

# Workflow flags
GENERATE=${GENERATE:-true}
PREP=${PREP:-true}
SIMULATE=${SIMULATE:-true}

# Create directories
mkdir -p logs pywrdrb/{inputs,outputs,models} figures

echo "========================================"
echo "Running climate-adjusted ensemble workflow"
echo "$np ranks on $SLURM_NNODES nodes"
echo "========================================"

# === climate_adjusted_low ===
echo ""
echo "========================================"
echo "Starting: climate_adjusted_low"
echo "========================================"

[ "$GENERATE" = true ] && {
    echo "Generating ensemble sets for climate_adjusted_low..."
    mpirun -np $np python3 01_generate_ensemble_sets.py "climate_adjusted_low"
}

[ "$PREP" = true ] && {
    echo "Preparing inputs for climate_adjusted_low..."
    mpirun -np $np python3 02_prep_pywrdrb_inputs.py "climate_adjusted_low"
}

[ "$SIMULATE" = true ] && {
    echo "Running simulations for climate_adjusted_low..."
    mpirun -np $np python3 03_run_pywrdrb_simulations.py "climate_adjusted_low"
}

echo "Completed: climate_adjusted_low"

# === climate_adjusted_high ===
echo ""
echo "========================================"
echo "Starting: climate_adjusted_high"
echo "========================================"

[ "$GENERATE" = true ] && {
    echo "Generating ensemble sets for climate_adjusted_high..."
    mpirun -np $np python3 01_generate_ensemble_sets.py "climate_adjusted_high"
}

[ "$PREP" = true ] && {
    echo "Preparing inputs for climate_adjusted_high..."
    mpirun -np $np python3 02_prep_pywrdrb_inputs.py "climate_adjusted_high"
}

[ "$SIMULATE" = true ] && {
    echo "Running simulations for climate_adjusted_high..."
    mpirun -np $np python3 03_run_pywrdrb_simulations.py "climate_adjusted_high"
}

echo "Completed: climate_adjusted_high"

echo ""
echo "========================================"
echo "All climate scenarios completed successfully!"
echo "========================================"
