#!/bin/bash
#SBATCH --job-name=perf
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

DATASET_ID="${1:?Usage: sbatch S5_calculate_performance_metrics_dataset.sh <dataset_id>}"

# make directories
mkdir -p logs figures

# Reconstruction has only 1 realization — use 1 rank
if [ "$DATASET_ID" = "reconstruction" ]; then
    np=1
fi

echo "========================================"
echo "Calculating performance metrics for $DATASET_ID with MPI ($np ranks)..."
echo "========================================"

mpirun -np $np python3 06_calculate_performance_metrics.py "$DATASET_ID" --all

if [ $? -ne 0 ]; then
    echo "ERROR: Performance metrics failed for $DATASET_ID"
    exit 1
fi

echo ""
echo "Performance metrics completed for $DATASET_ID!"
