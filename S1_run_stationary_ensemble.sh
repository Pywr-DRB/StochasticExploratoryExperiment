#!/bin/bash
#SBATCH --job-name=SA
#SBATCH --output=./logs/SA.out
#SBATCH --error=./logs/SA.err
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=40
#SBATCH --time=48:00:00
#SBATCH --mem=0
#SBATCH --exclusive

# Setup
DATASET_ID="${1:-stationary_ensemble}"  # Default or pass as argument
module load python/3.11.5
source venv/bin/activate
np=$(($SLURM_NTASKS_PER_NODE * $SLURM_NNODES))

# Workflow flags
GENERATE=${GENERATE:-false}
PREP=${PREP:-false}
SIMULATE=${SIMULATE:-true}

# Create directories
mkdir -p logs pywrdrb/{inputs,outputs,models} figures

echo "Running $DATASET_ID with $np ranks on $SLURM_NNODES nodes"

# Execute workflow
[ "$GENERATE" = true ] && mpirun -np $np python3 01_generate_ensemble_sets.py "$DATASET_ID"
[ "$PREP" = true ] && mpirun -np $np python3 02_prep_pywrdrb_inputs.py "$DATASET_ID"
[ "$SIMULATE" = true ] && mpirun -np $np python3 03_run_pywrdrb_simulations.py "$DATASET_ID"

echo "Workflow complete for $DATASET_ID"