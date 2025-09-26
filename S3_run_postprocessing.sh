#!/bin/bash
#SBATCH --job-name=post
#SBATCH --output=./logs/postprocessing.out
#SBATCH --error=./logs/postprocessing.err
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=40
#SBATCH --time=1:00:00
#SBATCH --mem=0
#SBATCH --exclusive


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


if [ "$CALCULATE_DROUGHT_METRICS" = true ]; then

    echo "Calculating SSI based drought metrics..."
    # mpirun -np $np python3 05_calculate_ssi_drought_metrics.py "stationary_ensemble"
    mpirun -np $np python3 05_calculate_ssi_drought_metrics.py "climate_adjusted_ssp245_min"
    mpirun -np $np python3 05_calculate_ssi_drought_metrics.py "climate_adjusted_ssp245_max"

    echo "Calculating Hashimoto metrics during droughts for stationary ensemble..."
    # mpirun -np $np python3 06_calculate_hashimoto_metrics_during_droughts.py "stationary_ensemble"
    mpirun -np $np python3 06_calculate_hashimoto_metrics_during_droughts.py "climate_adjusted_ssp245_min"
    mpirun -np $np python3 06_calculate_hashimoto_metrics_during_droughts.py "climate_adjusted_ssp245_max"

fi
