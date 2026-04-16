#!/bin/bash
#SBATCH --job-name=postprocess
#SBATCH --output=./logs/postprocess.out
#SBATCH --error=./logs/postprocess.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --exclusive

# Load modules and environment
module load python/3.11.5
source venv/bin/activate

# MPI transport: force libfabric TCP provider instead of RDMA verbs
export FI_PROVIDER=tcp

# Configuration name (determines output directory)
export CONFIG_NAME=${CONFIG_NAME:-default}

np=$(($SLURM_NTASKS_PER_NODE * $SLURM_NNODES))
USE_LOW_MEMORY=${USE_LOW_MEMORY:-true}

DATASETS=("stationary_ensemble" "climate_adjusted_low" "climate_adjusted_high")

mkdir -p logs

for DATASET_ID in "${DATASETS[@]}"; do
    echo "========================================"
    echo "Post-processing $DATASET_ID with MPI ($np ranks)..."
    echo "Low-memory mode: $USE_LOW_MEMORY"
    echo "========================================"

    if [ "$USE_LOW_MEMORY" = true ]; then
        mpirun -np $np python3 04_postprocess_data_mpi.py "$DATASET_ID" --low-memory --skip-recombine
    else
        mpirun -np $np python3 04_postprocess_data_mpi.py "$DATASET_ID"
    fi

    if [ $? -ne 0 ]; then
        echo "ERROR: Post-processing failed for $DATASET_ID"
        exit 1
    fi

    echo "Post-processing completed for $DATASET_ID!"
done

echo ""
echo "All datasets post-processed successfully."
