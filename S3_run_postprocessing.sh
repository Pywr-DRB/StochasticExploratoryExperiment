#!/bin/bash
#SBATCH --job-name=post
#SBATCH --output=./logs/postprocessing.out
#SBATCH --error=./logs/postprocessing.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0

# Load modules and environment
module load python/3.11.5
source venv/bin/activate
np=$(($SLURM_NTASKS_PER_NODE * $SLURM_NNODES))

# Workflow control flags 
RUN_POSTPROCESSING=${RUN_POSTPROCESSING:-true}
CALCULATE_STORAGE_ZONE_PROBABILITIES=${CALCULATE_STORAGE_ZONE_PROBABILITIES:-false}

# make directories
mkdir -p logs figures

if [ "$RUN_POSTPROCESSING" = true ]; then
    ################################################################################
    echo "Post-processing..."
    ################################################################################
    
    # Loop through datasets
    DATASETS=("stationary_ensemble" "climate_adjusted_low" "climate_adjusted_medium" "climate_adjusted_high")
    for DATASET_ID in "${DATASETS[@]}"; do
        echo "Post-processing $DATASET_ID..."
        python3 04_postprocess_data.py "$DATASET_ID"
    done
fi


if [ "$CALCULATE_STORAGE_ZONE_PROBABILITIES" = true ]; then
    ################################################################################
    echo "Calculating storage zone probabilities..."
    ################################################################################
    python3 06_calculate_storage_zone_probabilities.py --all
fi
