#!/bin/bash
#SBATCH --job-name=CAE
#SBATCH --output=./logs/CAE.out
#SBATCH --error=./logs/CAE.err
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=30

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

    # Execute workflow
    [ "$GENERATE" = true ] && mpirun -np $np python3 01_generate_ensemble_sets.py "$dataset"
    [ "$PREP" = true ] && mpirun -np $np python3 02_prep_pywrdrb_inputs.py "$dataset"
    [ "$SIMULATE" = true ] && mpirun -np $np python3 03_run_pywrdrb_simulations.py "$dataset"
    
echo "Completed generation->simulation for: $dataset"
done    

echo ""
echo "========================================"
echo "All climate scenarios completed successfully!"
echo "========================================"
