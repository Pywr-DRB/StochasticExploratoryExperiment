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

# MPI transport: use UCX (bypasses buggy OFI/libfabric RDMA path entirely)
export OMPI_MCA_pml=ucx
export OMPI_MCA_btl=self,vader
export OMPI_MCA_osc=ucx
export OMPI_MCA_mtl=^ofi,^psm2

np=$(($SLURM_NTASKS_PER_NODE * $SLURM_NNODES))

# Workflow control flags 
CALCULATE_DROUGHT_METRICS=${CALCULATE_DROUGHT_METRICS:-false}
CALCULATE_SATISFICING_DURING_DROUGHTS=${CALCULATE_SATISFICING_DURING_DROUGHTS:-false}
CALCULATE_EVENT_METRICS=${CALCULATE_EVENT_METRICS:-true}


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
    mpirun -np $np python3 06_calculate_satisficing_by_drought.py stationary_ensemble --all
    mpirun -np $np python3 06_calculate_satisficing_by_drought.py climate_adjusted_low --all
    mpirun -np $np python3 06_calculate_satisficing_by_drought.py climate_adjusted_high --all
fi



if [ "$CALCULATE_EVENT_METRICS" = true ]; then
    ################################################################################
    echo "Calculating per-drought-event metrics..."
    ################################################################################
    mpirun -np $np python3 07_calculate_event_metrics.py stationary_ensemble --all
    mpirun -np $np python3 07_calculate_event_metrics.py climate_adjusted_low --all
    mpirun -np $np python3 07_calculate_event_metrics.py climate_adjusted_high --all
fi