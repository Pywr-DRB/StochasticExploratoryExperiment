#!/bin/bash
#SBATCH --job-name=post
#SBATCH --output=./logs/post.out
#SBATCH --error=./logs/post.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=30
#SBATCH --mem-per-cpu=8G

# Load modules and environment
module load python/3.11.5
source venv/bin/activate

# MPI transport: force libfabric TCP provider instead of RDMA verbs
export FI_PROVIDER=tcp

# Calculate number of MPI ranks
# Optimal: N_RANKS = N_ENSEMBLE_SETS (currently 20 for 2000 realizations)
# Adjust --nodes and --ntasks-per-node above to match your N_ENSEMBLE_SETS
np=$(($SLURM_NTASKS_PER_NODE * $SLURM_NNODES))

# Workflow control flags
RUN_POSTPROCESSING=${RUN_POSTPROCESSING:-true}
CALCULATE_STORAGE_ZONE_PROBABILITIES=${CALCULATE_STORAGE_ZONE_PROBABILITIES:-false}

# Use low-memory mode to avoid MPI gather memory bottleneck
# Set to false for faster processing if memory is not a concern
USE_LOW_MEMORY=${USE_LOW_MEMORY:-true}

DATASETS=("stationary_ensemble" "climate_adjusted_low" "climate_adjusted_high")

# make directories
mkdir -p logs figures

if [ "$RUN_POSTPROCESSING" = true ]; then
    ################################################################################
    echo "MPI Parallel Post-processing..."
    echo "Number of MPI ranks: $np"
    echo "Low-memory mode: $USE_LOW_MEMORY"
    ################################################################################

    # Loop through datasets
    for DATASET_ID in "${DATASETS[@]}"; do
        echo ""
        echo "========================================"
        echo "Post-processing $DATASET_ID with MPI ($np ranks)..."
        echo "========================================"

        if [ "$USE_LOW_MEMORY" = true ]; then
            mpirun -np $np python3 04_postprocess_data_mpi.py "$DATASET_ID" --low-memory
        else
            mpirun -np $np python3 04_postprocess_data_mpi.py "$DATASET_ID"
        fi

        # Check exit status
        if [ $? -ne 0 ]; then
            echo "ERROR: Post-processing failed for $DATASET_ID"
            exit 1
        fi
    done

    echo ""
    echo "Post-processing completed for all datasets!"
fi



if [ "$CALCULATE_STORAGE_ZONE_PROBABILITIES" = true ]; then
    ################################################################################
    echo "Calculating storage zone probabilities..."
    ################################################################################
    python3 07_calculate_storage_zone_probabilities.py stationary_ensemble
    python3 07_calculate_storage_zone_probabilities.py climate_adjusted_low
    python3 07_calculate_storage_zone_probabilities.py climate_adjusted_high
fi
