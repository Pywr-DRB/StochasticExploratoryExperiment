#!/bin/bash
#SBATCH --job-name=perf
#SBATCH --output=./logs/perf.out
#SBATCH --error=./logs/perf.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --exclusive

# Load modules and environment
module load python/3.11.5
source venv/bin/activate

# MPI transport: force libfabric TCP provider instead of RDMA verbs
export FI_PROVIDER=tcp

np=$(($SLURM_NTASKS_PER_NODE * $SLURM_NNODES))

# make directories
mkdir -p logs figures

DATASETS=("stationary_ensemble" "climate_adjusted_low" "climate_adjusted_high")

for DATASET_ID in "${DATASETS[@]}"; do
    echo "========================================"
    echo "Calculating performance metrics for $DATASET_ID..."
    echo "========================================"
    mpirun -np $np python3 06_calculate_performance_metrics.py "$DATASET_ID"

    if [ $? -ne 0 ]; then
        echo "ERROR: Performance metrics failed for $DATASET_ID"
        exit 1
    fi
done

echo ""
echo "Performance metrics completed for all datasets!"
