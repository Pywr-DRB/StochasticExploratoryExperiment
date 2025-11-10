#!/bin/bash
#SBATCH --job-name=ssi
#SBATCH --output=./logs/ssi.out
#SBATCH --error=./logs/ssi.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0

# Load modules and environment
module load python/3.11.5
source venv/bin/activate
np=$(($SLURM_NTASKS_PER_NODE * $SLURM_NNODES))

# Workflow control flags 
CALCULATE_DROUGHT_METRICS=${CALCULATE_DROUGHT_METRICS:-true}
CALCULATE_SATISFICING_DURING_DROUGHTS=${CALCULATE_SATISFICING_DURING_DROUGHTS:-true}
TEST_CLUSTER_POTENTIAL=${TEST_CLUSTER_POTENTIAL:-false}


# make directories
mkdir -p logs figures


if [ "$CALCULATE_DROUGHT_METRICS" = true ]; then

    ################################################################################
    echo "Calculating SSI based drought metrics..."
    ################################################################################
    mpirun -np $np python3 05_calculate_ssi_drought_metrics.py "stationary_ensemble"
    mpirun -np $np python3 05_calculate_ssi_drought_metrics.py "climate_adjusted_low"
    mpirun -np $np python3 05_calculate_ssi_drought_metrics.py "climate_adjusted_high"

fi

if [ "$CALCULATE_SATISFICING_DURING_DROUGHTS" = true ]; then
    ################################################################################
    echo "Calculating satisficing during droughts..."
    ################################################################################
    python3 06_calculate_satisficing_by_drought.py "stationary_ensemble" 3
    python3 06_calculate_satisficing_by_drought.py "stationary_ensemble" 6
    python3 06_calculate_satisficing_by_drought.py "stationary_ensemble" 12
fi



if [ "$TEST_CLUSTER_POTENTIAL" = true ]; then

    ################################################################################
    echo "Testing cluster computing potential..."
    ################################################################################
    python3 SI3_evaluate_cluster_potential.py stationary_ensemble 3
    python3 SI3_evaluate_cluster_potential.py stationary_ensemble 6
    python3 SI3_evaluate_cluster_potential.py stationary_ensemble 12

fi