#!/bin/bash
#SBATCH --job-name=CAE
#SBATCH --output=./logs/CAE.out
#SBATCH --error=./logs/CAE.err
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=40
#SBATCH --time=144:00:00
#SBATCH --mem=0
#SBATCH --exclusive

# Setup
module load python/3.11.5
source venv/bin/activate
np=$(($SLURM_NTASKS_PER_NODE * $SLURM_NNODES))

# Workflow flags
GENERATE=${GENERATE:-false}
PREP=${PREP:-false}
SIMULATE=${SIMULATE:-true}

# Create directories
mkdir -p logs pywrdrb/{inputs,outputs,models} figures

# Climate scenarios to process
CLIMATE_SCENARIOS=(
    "climate_adjusted_low"
    "climate_adjusted_high"
)

echo "========================================"
echo "Running climate-adjusted ensemble workflow"
echo "Processing ${#CLIMATE_SCENARIOS[@]} scenarios with $np ranks on $SLURM_NNODES nodes"
echo "========================================"

# Loop through each climate scenario
for DATASET_ID in "${CLIMATE_SCENARIOS[@]}"; do
    echo ""
    echo "========================================"
    echo "Starting: $DATASET_ID"
    echo "========================================"

    # Execute workflow for this scenario
    [ "$GENERATE" = true ] && {
        echo "Generating ensemble sets for $DATASET_ID..."
        mpirun -np $np python3 01_generate_ensemble_sets.py "$DATASET_ID"
    }

    [ "$PREP" = true ] && {
        echo "Preparing inputs for $DATASET_ID..."
        mpirun -np $np python3 02_prep_pywrdrb_inputs.py "$DATASET_ID"
    }

    [ "$SIMULATE" = true ] && {
        echo "Running simulations for $DATASET_ID..."
        mpirun -np $np python3 03_run_pywrdrb_simulations.py "$DATASET_ID"
    }

    echo "Completed: $DATASET_ID"
done

echo ""
echo "========================================"
echo "All climate scenarios completed successfully!"
echo "========================================"
