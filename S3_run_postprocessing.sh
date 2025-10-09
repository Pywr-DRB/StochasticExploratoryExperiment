#!/bin/bash
#SBATCH --job-name=post
#SBATCH --output=./logs/postprocessing.out
#SBATCH --error=./logs/postprocessing.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --mem=0
#SBATCH --exclusive

# Load modules and environment
module load python/3.11.5
source venv/bin/activate
np=$(($SLURM_NTASKS_PER_NODE * $SLURM_NNODES))

# Workflow control flags 
CALCULATE_DROUGHT_METRICS=${CALCULATE_DROUGHT_METRICS:-false}
CALCULATE_STORAGE_ZONE_PROBABILITIES=${CALCULATE_STORAGE_ZONE_PROBABILITIES:-true}

# make directories
mkdir -p logs figures


if [ "$CALCULATE_DROUGHT_METRICS" = true ]; then

    ################################################################################
    echo "Calculating SSI based drought metrics..."
    ################################################################################
    # mpirun -np $np python3 05_calculate_ssi_drought_metrics.py "stationary_ensemble"
    # mpirun -np $np python3 05_calculate_ssi_drought_metrics.py "climate_adjusted_ssp245_min"
    # mpirun -np $np python3 05_calculate_ssi_drought_metrics.py "climate_adjusted_ssp245_max"
    # mpirun -np $np python3 05_calculate_ssi_drought_metrics.py "climate_adjusted_ssp245_median"
    mpirun -np $np python3 05_calculate_ssi_drought_metrics.py "climate_adjusted_ssp370_min"
    mpirun -np $np python3 05_calculate_ssi_drought_metrics.py "climate_adjusted_ssp370_max"
    mpirun -np $np python3 05_calculate_ssi_drought_metrics.py "climate_adjusted_ssp370_median"

fi

if [ "$CALCULATE_STORAGE_ZONE_PROBABILITIES" = true ]; then

    ################################################################################
    echo "Calculating storage zone probabilities..."
    ################################################################################
    python3 06_calculate_storage_zone_probabilities.py --all
    
fi
