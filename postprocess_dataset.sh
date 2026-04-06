#!/bin/bash
#SBATCH --job-name=post
#SBATCH --output=./logs/%x.out
#SBATCH --error=./logs/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --exclusive

# Load modules and environment
module load python/3.11.5
source venv/bin/activate

# MPI transport: force libfabric TCP provider instead of RDMA verbs
export FI_PROVIDER=tcp

# Optimal: N_RANKS = N_ENSEMBLE_SETS (currently 20 for 2000 realizations)
np=$(($SLURM_NTASKS_PER_NODE * $SLURM_NNODES))

# Use low-memory mode to avoid MPI gather memory bottleneck
USE_LOW_MEMORY=${USE_LOW_MEMORY:-true}

DATASET_ID="${1:?Usage: sbatch S3_postprocess_dataset.sh <dataset_id>}"

# make directories
mkdir -p logs figures

echo "========================================"
echo "Post-processing $DATASET_ID with MPI ($np ranks)..."
echo "Low-memory mode: $USE_LOW_MEMORY"
echo "========================================"

if [ "$USE_LOW_MEMORY" = true ]; then
    mpirun -np $np python3 04_postprocess_data_mpi.py "$DATASET_ID" --low-memory
else
    mpirun -np $np python3 04_postprocess_data_mpi.py "$DATASET_ID"
fi

if [ $? -ne 0 ]; then
    echo "ERROR: Post-processing failed for $DATASET_ID"
    exit 1
fi

echo ""
echo "Post-processing completed for $DATASET_ID!"
