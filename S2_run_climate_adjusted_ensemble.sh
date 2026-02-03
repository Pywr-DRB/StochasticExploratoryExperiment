#!/bin/bash
#SBATCH --job-name=CAE
#SBATCH --output=./logs/CAE.out
#SBATCH --error=./logs/CAE.err
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=30
#SBATCH --time=144:00:00
#SBATCH --exclusive
#SBATCH --mem=0

# Setup
module load python/3.11.5
source venv/bin/activate
np=$(($SLURM_NTASKS_PER_NODE * $SLURM_NNODES))

# Workflow flags
GENERATE=${GENERATE:-true}
PREP=${PREP:-true}
SIMULATE=${SIMULATE:-true}

DATASETS=("climate_adjusted_low" "climate_adjusted_high")

# Create directories
mkdir -p logs pywrdrb/{inputs,outputs,models} figures

echo "========================================"
echo "Running climate-adjusted ensemble workflow"
echo "$np ranks on $SLURM_NNODES nodes"
echo "========================================"

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "========================================"
    echo "Starting: $dataset"
    echo "========================================"
[ "$GENERATE" = true ] && {
    echo "Generating ensemble sets for $dataset..."
    mpirun -np $np python3 01_generate_ensemble_sets.py "$dataset"
}

[ "$PREP" = true ] && {
    echo "Preparing inputs for $dataset..."
    mpirun -np $np python3 02_prep_pywrdrb_inputs.py "$dataset"
}

[ "$SIMULATE" = true ] && {
    echo "Running simulations for $dataset..."
    mpirun -np $np python3 03_run_pywrdrb_simulations.py "$dataset"
}

echo "Completed: $dataset"
done    

echo ""
echo "========================================"
echo "All climate scenarios completed successfully!"
echo "========================================"
