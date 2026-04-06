#!/bin/bash
#SBATCH --job-name=ssi
#SBATCH --output=./logs/ssi.out
#SBATCH --error=./logs/ssi.err
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=20
#SBATCH --exclusive

# Load modules and environment
module load python/3.11.5
source venv/bin/activate

# MPI transport: force libfabric TCP provider instead of RDMA verbs
export FI_PROVIDER=tcp

# OpenMPI TCP tuning — reduce connection failures at scale
export OMPI_MCA_btl_tcp_links=1
export OMPI_MCA_mpi_yield_when_idle=1

# Configuration name (determines output directory)
export CONFIG_NAME=${CONFIG_NAME:-default}

np=$(($SLURM_NTASKS_PER_NODE * $SLURM_NNODES))

# make directories
mkdir -p logs

################################################################################
echo "Calculating SSI based drought metrics..."
################################################################################
python3 05_calculate_ssi_drought_metrics.py historic
mpirun -np $np python3 05_calculate_ssi_drought_metrics.py stationary_ensemble
mpirun -np $np python3 05_calculate_ssi_drought_metrics.py climate_adjusted_low
mpirun -np $np python3 05_calculate_ssi_drought_metrics.py climate_adjusted_high
