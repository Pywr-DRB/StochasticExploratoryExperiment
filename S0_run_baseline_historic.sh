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

# MPI transport configuration for OpenMPI 4.0.5 + libfabric stability
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_mtl=^ofi

# Create directories
mkdir -p logs pywrdrb/{inputs,outputs,models} figures

echo "Running $DATASET_ID with $np ranks on $SLURM_NNODES nodes"

# Execute workflow
srun python3 00_run_baseline_simulations.py

echo "Done simulating historic baselines."