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
RUN_POSTPROCESSING=${RUN_POSTPROCESSING:-false}
CALCULATE_STORAGE_ZONE_PROBABILITIES=${CALCULATE_STORAGE_ZONE_PROBABILITIES:-false}
CALCULATE_SATISFICING_DURING_DROUGHTS=${CALCULATE_SATISFICING_DURING_DROUGHTS:-true}

DATASETS=("stationary_ensemble" "climate_adjusted_low" "climate_adjusted_medium" "climate_adjusted_high")

# make directories
mkdir -p logs figures

if [ "$RUN_POSTPROCESSING" = true ]; then
    ################################################################################
    echo "Post-processing..."
    ################################################################################
    
    # Loop through datasets
    for DATASET_ID in "${DATASETS[@]}"; do
        echo "Post-processing $DATASET_ID..."
        python3 04_postprocess_data.py "$DATASET_ID"
    done
fi


if [ "$CALCULATE_SATISFICING_DURING_DROUGHTS" = true ]; then
    ################################################################################
    echo "Calculating satisficing during droughts..."
    ################################################################################
    python3 06_calculate_satisficing_by_drought.py "stationary_ensemble" 3
    python3 06_calculate_satisficing_by_drought.py "stationary_ensemble" 6
    python3 06_calculate_satisficing_by_drought.py "stationary_ensemble" 12
fi


if [ "$CALCULATE_STORAGE_ZONE_PROBABILITIES" = true ]; then
    ################################################################################
    echo "Calculating storage zone probabilities..."
    ################################################################################
    python3 07_calculate_storage_zone_probabilities.py --all
fi
