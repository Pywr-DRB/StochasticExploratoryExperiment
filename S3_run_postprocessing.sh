#!/bin/bash
#SBATCH --job-name=post
#SBATCH --output=./logs/post.out
#SBATCH --error=./logs/post.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=30

# Load modules and environment
module load python/3.11.5
source venv/bin/activate

# MPI transport: force libfabric TCP provider instead of RDMA verbs
export FI_PROVIDER=tcp

# Workflow control flags
RUN_POSTPROCESSING=${RUN_POSTPROCESSING:-true}
CALCULATE_STORAGE_ZONE_PROBABILITIES=${CALCULATE_STORAGE_ZONE_PROBABILITIES:-false}

# Use low-memory mode to avoid MPI gather memory bottleneck
# Set to false for faster processing if memory is not a concern
USE_LOW_MEMORY=${USE_LOW_MEMORY:-true}

# Each dataset needs N_ENSEMBLE_SETS ranks (currently 20).
# With 3 datasets running concurrently: 3 x 20 = 60 ranks total.
RANKS_PER_DATASET=20
DATASETS=("stationary_ensemble" "climate_adjusted_low" "climate_adjusted_high")

# make directories
mkdir -p logs figures

if [ "$RUN_POSTPROCESSING" = true ]; then
    ################################################################################
    echo "MPI Parallel Post-processing (3 datasets concurrently)..."
    echo "Ranks per dataset: $RANKS_PER_DATASET"
    echo "Low-memory mode: $USE_LOW_MEMORY"
    ################################################################################

    LOW_MEM_FLAG=""
    if [ "$USE_LOW_MEMORY" = true ]; then
        LOW_MEM_FLAG="--low-memory"
    fi

    # Launch all 3 datasets concurrently as separate SLURM job steps.
    # srun --exact partitions the allocation so each step gets its own ranks.
    PIDS=()
    for DATASET_ID in "${DATASETS[@]}"; do
        echo "Launching $DATASET_ID ($RANKS_PER_DATASET ranks)..."
        srun --exact -n $RANKS_PER_DATASET --output="./logs/post_${DATASET_ID}.out" \
            --error="./logs/post_${DATASET_ID}.err" \
            python3 04_postprocess_data_mpi.py "$DATASET_ID" $LOW_MEM_FLAG &
        PIDS+=($!)
    done

    # Wait for all and check exit status
    FAIL=0
    for i in "${!DATASETS[@]}"; do
        wait "${PIDS[$i]}"
        STATUS=$?
        if [ $STATUS -ne 0 ]; then
            echo "ERROR: Post-processing failed for ${DATASETS[$i]} (exit $STATUS)"
            echo "  See logs/post_${DATASETS[$i]}.err for details"
            FAIL=1
        else
            echo "SUCCESS: ${DATASETS[$i]} completed"
        fi
    done

    if [ $FAIL -ne 0 ]; then
        exit 1
    fi

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
