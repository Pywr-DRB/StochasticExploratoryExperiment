#!/bin/bash
# Submit the entire workflow as a chain of SLURM jobs with dependencies.
#
# Execution order:
#   S0  (baseline)
#   S1 & S2  (stationary + climate-adjusted ensembles, in parallel)
#   S3  (postprocess all 3 datasets, in parallel)
#   S4  (SSI drought metrics)
#   S5 & S6  (figures + SI figures, in parallel)
#
# Usage: bash S99_run_entire_workflow.sh

set -e
mkdir -p logs

echo "============================================================"
echo "SUBMITTING FULL WORKFLOW"
echo "============================================================"

# --- S0: Baseline historic simulations ---
S0=$(sbatch --parsable S0_run_baseline_historic.sh)
echo "S0 baseline:         job $S0"

# --- S1 & S2: Ensemble generation (parallel, after S0) ---
S1=$(sbatch --parsable --dependency=afterok:$S0 S1_run_stationary_ensemble.sh)
echo "S1 stationary:       job $S1 (after S0)"

S2=$(sbatch --parsable --dependency=afterok:$S0 S2_run_climate_adjusted_ensemble.sh)
echo "S2 climate-adjusted: job $S2 (after S0)"

# --- S3: Postprocess each dataset (parallel, after S1 & S2) ---
DATASETS=("stationary_ensemble" "climate_adjusted_low" "climate_adjusted_high")
S3_IDS=()
for DATASET_ID in "${DATASETS[@]}"; do
    JID=$(sbatch --parsable --dependency=afterok:$S1:$S2 \
        --job-name="post_${DATASET_ID}" \
        S3_postprocess_dataset.sh "$DATASET_ID")
    S3_IDS+=($JID)
    echo "S3 post $DATASET_ID: job $JID (after S1,S2)"
done

S3_DEP=$(IFS=:; echo "${S3_IDS[*]}")

# --- S4: SSI drought metrics (after all S3 jobs) ---
S4=$(sbatch --parsable --dependency=afterok:$S3_DEP S4_calculate_ssi.sh)
echo "S4 SSI metrics:      job $S4 (after S3)"

# --- S5 & S6: Figures (parallel, after S4) ---
S5=$(sbatch --parsable --dependency=afterok:$S4 S5_run_figure_generation.sh)
echo "S5 figures:          job $S5 (after S4)"

S6=$(sbatch --parsable --dependency=afterok:$S4 S6_run_SI_scripts.sh)
echo "S6 SI figures:       job $S6 (after S4)"

echo ""
echo "============================================================"
echo "ALL JOBS SUBMITTED"
echo "============================================================"
echo "Monitor with: squeue -u \$USER"
echo "Cancel all:   scancel $S0 $S1 $S2 ${S3_IDS[*]} $S4 $S5 $S6"
