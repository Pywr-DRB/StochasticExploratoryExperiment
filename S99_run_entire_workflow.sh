#!/bin/bash
# Submit the full workflow as a chain of dependent SLURM jobs.
#
#   S0        baseline reconstruction                   1 node
#   S1_gen    stationary ensemble: generate (01)        8 nodes
#   S1_prep   stationary ensemble: prep (02)            1 node x 40 ranks
#   S1_sim    stationary ensemble: simulate (03)        8 nodes
#   S2_gen    climate-adjusted (low, high): generate    5 nodes
#   S2_prep   climate-adjusted: prep                    1 node x 40 ranks
#   S2_sim    climate-adjusted: simulate                5 nodes
#   S3 -> S4 -> S5 -> {S6, S7, S8}
#
# The prep steps run on a single node because the Pywr-DRB v2.2 ensemble
# preprocessors scatter data from rank 0 in a way that stalls over
# cross-node TCP at scale; on one node all MPI traffic is shared-memory.
#
# Usage: bash S99_run_entire_workflow.sh

set -e

mkdir -p logs

echo "============================================================"
echo "SUBMITTING FULL WORKFLOW"
echo "============================================================"

# --- S0: Baseline historic simulations ---
S0=$(sbatch --parsable S0_run_baseline_historic.sh)
echo "S0 baseline:          job $S0"

# --- S1: Stationary ensemble, split into gen / prep / sim ---
S1G=$(sbatch --parsable --job-name=SA_gen  --output=./logs/SA_gen.out  --error=./logs/SA_gen.err \
      --dependency=afterok:$S0 --export=ALL,PREP=false,SIMULATE=false S1_run_stationary_ensemble.sh)
echo "S1 gen (8 nodes):     job $S1G (after S0)"

S1P=$(sbatch --parsable --job-name=SA_prep --output=./logs/SA_prep.out --error=./logs/SA_prep.err \
      --nodes=1 --ntasks-per-node=40 --mem=0 \
      --dependency=afterok:$S1G --export=ALL,GENERATE=false,SIMULATE=false S1_run_stationary_ensemble.sh)
echo "S1 prep (1 node):     job $S1P (after S1 gen)"

# S2 gen runs concurrently with S1 prep (5 + 1 nodes <= 8)
S2G=$(sbatch --parsable --job-name=CAE_gen --output=./logs/CAE_gen.out --error=./logs/CAE_gen.err \
      --dependency=afterok:$S1G --export=ALL,PREP=false,SIMULATE=false S2_run_climate_adjusted_ensemble.sh)
echo "S2 gen (5 nodes):     job $S2G (after S1 gen; concurrent with S1 prep)"

S1S=$(sbatch --parsable --job-name=SA_sim  --output=./logs/SA_sim.out  --error=./logs/SA_sim.err \
      --dependency=afterok:$S1P:$S2G --export=ALL,GENERATE=false,PREP=false S1_run_stationary_ensemble.sh)
echo "S1 sim (8 nodes):     job $S1S (after S1 prep, S2 gen)"

# --- S2: Climate-adjusted ensembles, prep / sim ---
S2P=$(sbatch --parsable --job-name=CAE_prep --output=./logs/CAE_prep.out --error=./logs/CAE_prep.err \
      --nodes=1 --ntasks-per-node=40 --mem=0 \
      --dependency=afterok:$S1S --export=ALL,GENERATE=false,SIMULATE=false S2_run_climate_adjusted_ensemble.sh)
echo "S2 prep (1 node):     job $S2P (after S1 sim)"

S2S=$(sbatch --parsable --job-name=CAE_sim --output=./logs/CAE_sim.out --error=./logs/CAE_sim.err \
      --dependency=afterok:$S2P --export=ALL,GENERATE=false,PREP=false S2_run_climate_adjusted_ensemble.sh)
echo "S2 sim (5 nodes):     job $S2S (after S2 prep)"

# --- S3: Postprocess all datasets ---
S3=$(sbatch --parsable --dependency=afterok:$S1S:$S2S S3_postprocess_all.sh)
echo "S3 postprocess:       job $S3 (after S1 sim, S2 sim)"

# --- S4: SSI drought metrics ---
S4=$(sbatch --parsable --dependency=afterok:$S3 S4_calculate_ssi.sh)
echo "S4 SSI metrics:       job $S4 (after S3)"

# --- S5: Performance metrics + storage zone probabilities ---
S5=$(sbatch --parsable --dependency=afterok:$S4 S5_calculate_performance_metrics.sh)
echo "S5 perf metrics:      job $S5 (after S4)"

# --- S6, S7, S8: figures, SI figures, manuscript values (parallel) ---
S6=$(sbatch --parsable --dependency=afterok:$S5 S6_run_figure_generation.sh)
echo "S6 figures:           job $S6 (after S5)"
S7=$(sbatch --parsable --dependency=afterok:$S5 S7_run_SI_scripts.sh)
echo "S7 SI figures:        job $S7 (after S5)"
S8=$(sbatch --parsable --dependency=afterok:$S5 S8_extract_manuscript_values.sh)
echo "S8 manuscript vals:   job $S8 (after S5)"

echo ""
echo "============================================================"
echo "ALL JOBS SUBMITTED"
echo "============================================================"
echo "Monitor with: squeue -u \$USER"
echo "Cancel all:   scancel $S0 $S1G $S1P $S2G $S1S $S2P $S2S $S3 $S4 $S5 $S6 $S7 $S8"
