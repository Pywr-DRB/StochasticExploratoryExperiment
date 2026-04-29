#!/bin/bash
# Submit the entire workflow as a chain of SLURM jobs with dependencies.
#
# Execution order:
#   S0  (baseline)
#   S1 & S2  (stationary + climate-adjusted ensembles, in parallel)
#   S3  (postprocess all 3 datasets)
#   S4  (SSI drought metrics)
#   S5  (performance metrics, all 3 datasets in parallel)
#   S6 & S7 & S8  (figures, SI figures, manuscript value extraction — in parallel)
#
# Usage: bash S99_run_entire_workflow.sh

set -e

# Configuration name (determines output directory)
export CONFIG_NAME=${CONFIG_NAME:-default}

mkdir -p logs

echo "============================================================"
echo "SUBMITTING FULL WORKFLOW"
echo "============================================================"

# # --- S0: Baseline historic simulations ---
S0=$(sbatch --parsable S0_run_baseline_historic.sh)
echo "S0 baseline:         job $S0"

# --- S1 & S2: Ensemble generation (parallel, after S0) ---
S1=$(sbatch --parsable --dependency=afterok:$S0 S1_run_stationary_ensemble.sh)
echo "S1 stationary:       job $S1 (after S0)"

S2=$(sbatch --parsable --dependency=afterok:$S0 S2_run_climate_adjusted_ensemble.sh)
echo "S2 climate-adjusted: job $S2 (after S0)"

# --- S3: Postprocess each dataset (after S1 & S2) ---
S3=$(sbatch --parsable --dependency=afterok:$S1:$S2 S3_postprocess_all.sh)
echo "S3 postprocess:      job $S3 (after S1,S2)"

# --- S4: SSI drought metrics (after S3) ---
S4=$(sbatch --parsable --dependency=afterok:$S3 S4_calculate_ssi.sh)
echo "S4 SSI metrics:      job $S4 (after S3)"


# --- S5: Performance metrics per dataset (parallel, after all S4 jobs) ---
S5=$(sbatch --parsable --dependency=afterok:$S4 S5_calculate_performance_metrics.sh)
echo "S5 performance metrics:      job $S5 (after S4)"


S5_DEP=$S5

# --- S6 & S7: Figures (parallel, after all S5 jobs) ---
S6=$(sbatch --parsable --dependency=afterok:$S5_DEP S6_run_figure_generation.sh)
echo "S6 figures:          job $S6 (after S5)"

S7=$(sbatch --parsable --dependency=afterok:$S5_DEP S7_run_SI_scripts.sh)
echo "S7 SI figures:       job $S7 (after S5)"

S8=$(sbatch --parsable --dependency=afterok:$S5_DEP S8_extract_manuscript_values.sh)
echo "S8 manuscript vals:  job $S8 (after S5)"

echo ""
echo "============================================================"
echo "ALL JOBS SUBMITTED"
echo "============================================================"
echo "Monitor with: squeue -u \$USER"
echo "Cancel all:   scancel $S0 $S1 $S2 $S3 $S4 $S5 $S6 $S7 $S8"
