#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --output=./logs/baseline.out
#SBATCH --error=./logs/baseline.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=15
#SBATCH --time=48:00:00
#SBATCH --mem=0

# Setup
module load python/3.11.5
source venv/bin/activate
np=$(($SLURM_NTASKS_PER_NODE * $SLURM_NNODES))

# MPI transport: force libfabric TCP provider instead of RDMA verbs
export FI_PROVIDER=tcp

# Configuration name (determines output directory)
export CONFIG_NAME=${CONFIG_NAME:-default}

# Create directories (output dirs created by Python via ensure_ensemble_set_dirs)
mkdir -p logs pywrdrb/inputs

echo "Running $DATASET_ID with $np ranks on $SLURM_NNODES nodes"

# Execute workflow
mpirun -np $np python3 00_run_baseline_simulations.py

echo "Done simulating historic baselines."