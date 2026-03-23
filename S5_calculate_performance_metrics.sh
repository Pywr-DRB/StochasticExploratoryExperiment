#!/bin/bash
# Submit performance metric jobs for all datasets in parallel.
# Each dataset gets its own SLURM job with 20 MPI ranks.
#
# Usage: bash S5_calculate_performance_metrics.sh

DATASETS=("stationary_ensemble" "climate_adjusted_low" "climate_adjusted_high" "reconstruction")

mkdir -p logs

for DATASET_ID in "${DATASETS[@]}"; do
    JOB_ID=$(sbatch --job-name="perf_${DATASET_ID}" --parsable \
        S5_calculate_performance_metrics_dataset.sh "$DATASET_ID")
    echo "Submitted $DATASET_ID: job $JOB_ID"
done
