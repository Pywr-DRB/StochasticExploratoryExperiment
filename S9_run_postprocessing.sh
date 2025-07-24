#!/bin/bash
#SBATCH --job-name=post
#SBATCH --output=./logs/postprocessing.out
#SBATCH --error=./logs/postprocessing.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem=0

# =============================================================================
# SETUP
# =============================================================================

# Load modules and environment
module load python/3.11.5
source venv/bin/activate
np=$(($SLURM_NTASKS_PER_NODE * $SLURM_NNODES))

# Read configuration from Python config file
eval $(python3 -c "from config import *")

# Workflow control flags 
CALCULATE_DROUGHT_METRICS=${CALCULATE_DROUGHT_METRICS:-true}

# make directories
mkdir -p logs figures

# =============================================================================
# WORKFLOW
# =============================================================================

echo "========================================================================"
echo "POSTPROCESSING ENSEMBLE WORKFLOW STARTING"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total MPI ranks: $np"
echo "Total realizations: $TOTAL_REALIZATIONS"
echo "Ensemble sets: $N_ENSEMBLE_SETS"
echo "Realizations per set: $N_REALIZATIONS_PER_ENSEMBLE_SET"
echo "========================================================================"



# Step 6: Analyze outcomes and generate plots (single core)
if [ "$CALCULATE_DROUGHT_METRICS" = true ]; then
    
    # echo "Calculating SSI based drought metrics for stationary ensemble..."
    # python3 05_calculate_ssi_drought_metrics.py "stationary"

    # echo "Calculating shortages for stationary ensemble..."
    # python3 06_calculate_shortages.py "stationary"

    # echo "Calculating Hashimoto metrics during droughts for stationary ensemble..."
    # python3 07_calculate_hashimoto_metrics_during_droughts.py "stationary"

    # echo "Plotting stationary drought scatters..."
    # python3 999_plot_drought_montague_violations.py "stationary"

    echo "----------------------------------------"

    # # # Repeat for climate adjusted ensemble
    # echo "Calculating SSI based drought metrics for climate adjusted ensemble..."
    # python3 05_calculate_ssi_drought_metrics.py "climate_adjusted"

    # echo "Calculating shortages for climate adjusted ensemble..."
    # python3 06_calculate_shortages.py "climate_adjusted"

    echo "Calculating Hashimoto metrics during droughts for climate adjusted ensemble..."
    python3 07_calculate_hashimoto_metrics_during_droughts.py "climate_adjusted"

    echo "Plotting..."
    python3 999_plot_drought_montague_violations.py "climate_adjusted"

    echo "----------------------------------------"
fi


echo "========================================================================"
echo "POST-PROCESSING COMPLETE - CHECK LOGS FOR DETAILED RESULTS"
echo "========================================================================"