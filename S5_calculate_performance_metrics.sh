#!/bin/bash
#SBATCH --job-name=perf_metrics
#SBATCH --output=./logs/perf_metrics.out
#SBATCH --error=./logs/perf_metrics.err
#SBATCH --nodes=2
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

DATASETS=("stationary_ensemble" "climate_adjusted_low" "climate_adjusted_high" "reconstruction")

mkdir -p logs

for DATASET_ID in "${DATASETS[@]}"; do
    # Reconstruction has only 1 realization — use 1 rank
    if [ "$DATASET_ID" = "reconstruction" ]; then
        RUN_NP=1
    else
        RUN_NP=$np
    fi

    echo "========================================"
    echo "Calculating performance metrics for $DATASET_ID with MPI ($RUN_NP ranks)..."
    echo "========================================"

    mpirun -np $RUN_NP python3 06_calculate_performance_metrics.py "$DATASET_ID" --all

    if [ $? -ne 0 ]; then
        echo "ERROR: Performance metrics failed for $DATASET_ID"
        exit 1
    fi

    echo "Performance metrics completed for $DATASET_ID!"
done

echo ""
echo "All performance metrics processed successfully."

################################################################################
echo "========================================"
echo "Calculating storage zone probabilities..."
echo "========================================"
python3 si_scripts/SI3_calculate_storage_zone_probabilities.py --all

if [ $? -ne 0 ]; then
    echo "ERROR: Zone probability calculation failed"
    exit 1
fi

echo ""
echo "All metrics and zone probabilities completed successfully."
