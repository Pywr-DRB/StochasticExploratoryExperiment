#!/bin/bash
#SBATCH --job-name=SA
#SBATCH --output=./logs/SA.out
#SBATCH --error=./logs/SA.err
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=20
#SBATCH --exclusive

# Setup
module load python/3.11.5
source venv/bin/activate
np=$(($SLURM_NTASKS_PER_NODE * $SLURM_NNODES))

# MPI transport: force libfabric TCP provider instead of RDMA verbs
# (OpenMPI 4.0.5 + libfabric 1.12.1 verbs/RDMA crashes at scale)
export FI_PROVIDER=tcp

# OpenMPI TCP tuning — reduce connection failures at scale
export OMPI_MCA_btl_tcp_links=1
export OMPI_MCA_mpi_yield_when_idle=1

# Configuration name (determines output directory)
export CONFIG_NAME=${CONFIG_NAME:-default}

# Workflow flags
DATASET_ID="${1:-stationary_ensemble}"
GENERATE=${GENERATE:-true}
PREP=${PREP:-true}
SIMULATE=${SIMULATE:-true}

# Create directories
mkdir -p logs pywrdrb/inputs

echo "Running $DATASET_ID with $np ranks on $SLURM_NNODES nodes"

# Execute workflow
[ "$GENERATE" = true ] && mpirun -np $np python3 01_generate_ensemble_sets.py "$DATASET_ID"
[ "$PREP" = true ] && mpirun -np $np python3 02_prep_pywrdrb_inputs.py "$DATASET_ID"
[ "$SIMULATE" = true ] && mpirun -np $np python3 03_run_pywrdrb_simulations.py "$DATASET_ID"

echo "Workflow complete for $DATASET_ID"