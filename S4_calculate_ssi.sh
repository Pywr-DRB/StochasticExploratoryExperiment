#!/bin/bash
#SBATCH --job-name=ssi
#SBATCH --output=./logs/ssi.out
#SBATCH --error=./logs/ssi.err
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=40
#SBATCH --mem-per-cpu=8G

# Load modules and environment
module load python/3.11.5
source venv/bin/activate

# MPI transport: force libfabric TCP provider instead of RDMA verbs
export FI_PROVIDER=tcp

np=$(($SLURM_NTASKS_PER_NODE * $SLURM_NNODES))

# Workflow control flags
CALCULATE_DROUGHT_METRICS=${CALCULATE_DROUGHT_METRICS:-false}
CALCULATE_DROUGHT_ANALYSIS=${CALCULATE_DROUGHT_ANALYSIS:-true}

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

if [ "$CALCULATE_DROUGHT_ANALYSIS" = true ]; then
    ################################################################################
    echo "Calculating annual satisficing & per-event metrics..."
    ################################################################################
    mpirun -np $np python3 06_calculate_drought_analysis.py stationary_ensemble --all
    mpirun -np $np python3 06_calculate_drought_analysis.py climate_adjusted_low --all
    mpirun -np $np python3 06_calculate_drought_analysis.py climate_adjusted_high --all
fi
