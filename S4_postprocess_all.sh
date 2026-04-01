#!/bin/bash
# Submit postprocessing jobs for all datasets in parallel.
# Each dataset gets its own SLURM job with 20 MPI ranks.
#
# Usage: bash S4_postprocess_all.sh

DATASETS=("stationary_ensemble" "climate_adjusted_low" "climate_adjusted_high")

mkdir -p logs

for DATASET_ID in "${DATASETS[@]}"; do
    JOB_ID=$(sbatch --job-name="post_${DATASET_ID}" --parsable \
        postprocess_dataset.sh "$DATASET_ID")
    echo "Submitted $DATASET_ID: job $JOB_ID"
done
