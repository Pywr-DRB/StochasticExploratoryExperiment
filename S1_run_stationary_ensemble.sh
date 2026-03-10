#!/bin/bash
#SBATCH --job-name=SA
#SBATCH --output=./logs/SA.out
#SBATCH --error=./logs/SA.err
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=30
#SBATCH --mem-per-cpu=8G

# Setup
module load python/3.11.5
source venv/bin/activate
np=$(($SLURM_NTASKS_PER_NODE * $SLURM_NNODES))

# MPI transport configuration for OpenMPI 4.0.5 + libfabric stability
# Use ob1 PML with TCP + shared-memory transports (bypass RDMA/OFI crashes)
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_mtl=^ofi

# Workflow flags
DATASET_ID="${1:-stationary_ensemble}"
GENERATE=${GENERATE:-true}
PREP=${PREP:-true}
SIMULATE=${SIMULATE:-true}

# Create directories
mkdir -p logs pywrdrb/{inputs,outputs,models} figures

echo "Running $DATASET_ID with $np ranks on $SLURM_NNODES nodes"

# Execute workflow
[ "$GENERATE" = true ] && srun python3 01_generate_ensemble_sets.py "$DATASET_ID"
[ "$PREP" = true ] && srun python3 02_prep_pywrdrb_inputs.py "$DATASET_ID"
[ "$SIMULATE" = true ] && srun python3 03_run_pywrdrb_simulations.py "$DATASET_ID"

echo "Workflow complete for $DATASET_ID"