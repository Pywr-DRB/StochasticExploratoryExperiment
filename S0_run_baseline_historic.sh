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

# MPI transport: use UCX (bypasses buggy OFI/libfabric RDMA path entirely)
export OMPI_MCA_pml=ucx
export OMPI_MCA_btl=self,vader
export OMPI_MCA_osc=ucx
export OMPI_MCA_mtl=^ofi,^psm2

# Create directories
mkdir -p logs pywrdrb/{inputs,outputs,models} figures

echo "Running $DATASET_ID with $np ranks on $SLURM_NNODES nodes"

# Execute workflow
mpirun -np $np python3 00_run_baseline_simulations.py

echo "Done simulating historic baselines."