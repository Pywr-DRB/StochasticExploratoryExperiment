#!/bin/bash
#SBATCH --job-name=ssi
#SBATCH --output=./logs/ssi.out
#SBATCH --error=./logs/ssi.err
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=40
#SBATCH --mem=0

# Load modules and environment
module load python/3.11.5
source venv/bin/activate
np=$(($SLURM_NTASKS_PER_NODE * $SLURM_NNODES))

# Workflow control flags 
CALCULATE_DROUGHT_METRICS=${CALCULATE_DROUGHT_METRICS:-true}
CALCULATE_SATISFICING_DURING_DROUGHTS=${CALCULATE_SATISFICING_DURING_DROUGHTS:-false}


# make directories
mkdir -p logs figures


if [ "$CALCULATE_DROUGHT_METRICS" = true ]; then

    ################################################################################
    echo "Calculating SSI based drought metrics..."
    ################################################################################
    python3 05_calculate_ssi_drought_metrics.py historic
    mpirun -np $np python3 05_calculate_ssi_drought_metrics.py stationary_ensemble
    mpirun -np $np python3 05_calculate_ssi_drought_metrics.py climate_adjusted_low
    mpirun -np $np python3 05_calculate_ssi_drought_metrics.py climate_adjusted_high
    

fi

if [ "$CALCULATE_SATISFICING_DURING_DROUGHTS" = true ]; then
    ################################################################################
    echo "Calculating satisficing during droughts..."
    ################################################################################
    python3 06_calculate_satisficing_by_drought.py stationary_ensemble --all
    python3 06_calculate_satisficing_by_drought.py climate_adjusted_low --all
    python3 06_calculate_satisficing_by_drought.py climate_adjusted_high --all
fi