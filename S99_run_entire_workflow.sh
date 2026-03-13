#!/bin/bash
# Submit the entire workflow as a chain of SLURM jobs with dependencies.
#
# Execution order:
#   S0  (baseline)
#   S1 & S2  (stationary + climate-adjusted ensembles, in parallel)
#   S3  (SSI drought metrics)
#   S4  (postprocess all 3 datasets, in parallel)
#   S5  (performance metrics, all 3 datasets in parallel)
#   S6 & S7  (figures + SI figures, in parallel)
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

# --- S3: SSI drought metrics (after S1 & S2) ---
S3=$(sbatch --parsable --dependency=afterok:$S1:$S2 S3_calculate_ssi.sh)
echo "S3 SSI metrics:      job $S3 (after S1,S2)"

# --- S4: Postprocess each dataset (parallel, after S3) ---
DATASETS=("stationary_ensemble" "climate_adjusted_low" "climate_adjusted_high")
S4_IDS=()
for DATASET_ID in "${DATASETS[@]}"; do
    JID=$(sbatch --parsable --dependency=afterok:$S3 \
        --job-name="post_${DATASET_ID}" \
        S4_postprocess_dataset.sh "$DATASET_ID")
    S4_IDS+=($JID)
    echo "S4 post $DATASET_ID: job $JID (after S3)"
done

S4_DEP=$(IFS=:; echo "${S4_IDS[*]}")

# --- S5: Performance metrics per dataset (parallel, after all S4 jobs) ---
S5_IDS=()
for DATASET_ID in "${DATASETS[@]}"; do
    JID=$(sbatch --parsable --dependency=afterok:$S4_DEP \
        --job-name="perf_${DATASET_ID}" \
        S5_calculate_performance_metrics_dataset.sh "$DATASET_ID")
    S5_IDS+=($JID)
    echo "S5 perf $DATASET_ID: job $JID (after S4)"
done

S5_DEP=$(IFS=:; echo "${S5_IDS[*]}")

# --- S6 & S7: Figures (parallel, after all S5 jobs) ---
S6=$(sbatch --parsable --dependency=afterok:$S5_DEP S6_run_figure_generation.sh)
echo "S6 figures:          job $S6 (after S5)"

S7=$(sbatch --parsable --dependency=afterok:$S5_DEP S7_run_SI_scripts.sh)
echo "S7 SI figures:       job $S7 (after S5)"

echo ""
echo "============================================================"
echo "ALL JOBS SUBMITTED"
echo "============================================================"
echo "Monitor with: squeue -u \$USER"
echo "Cancel all:   scancel $S0 $S1 $S2 $S3 ${S4_IDS[*]} ${S5_IDS[*]} $S6 $S7"
