#!/bin/bash
#SBATCH --job-name=CAE
#SBATCH --output=./logs/CAE.out
#SBATCH --error=./logs/CAE.err
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=20
#SBATCH --exclusive

# Setup
module load python/3.11.5
source venv/bin/activate
np=$(($SLURM_NTASKS_PER_NODE * $SLURM_NNODES))

# MPI transport: pin TCP traffic to the InfiniBand fabric (ib0).
#   - btl_tcp_if_include: rank-to-rank TCP data channel
#   - oob_tcp_if_include: out-of-band launch/wireup channel
# Prevents OpenMPI from advertising the public NIC (eno1np0) or iDRAC mgmt
# interface and stalling connect() at scale.
export OMPI_MCA_btl_tcp_if_include=ib0
export OMPI_MCA_oob_tcp_if_include=ib0

# Disable the openib/verbs BTL so OpenMPI does not probe RDMA paths.
export OMPI_MCA_btl=^openib

# OpenMPI TCP tuning — reduce connection failures at scale
export OMPI_MCA_btl_tcp_links=1
export OMPI_MCA_mpi_yield_when_idle=1

# Configuration name (determines output directory)
export CONFIG_NAME=${CONFIG_NAME:-default}

# Workflow flags
GENERATE=${GENERATE:-true}
PREP=${PREP:-true}
SIMULATE=${SIMULATE:-true}

DATASETS=("climate_adjusted_low" "climate_adjusted_high")

# Create directories
mkdir -p logs pywrdrb/inputs

echo "========================================"
echo "Running climate-adjusted ensemble workflow"
echo "$np ranks on $SLURM_NNODES nodes"
echo "========================================"

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "========================================"
    echo "Starting: $dataset"
    echo "========================================"

    # Execute workflow
    [ "$GENERATE" = true ] && mpirun -np $np python3 01_generate_ensemble_sets.py "$dataset"
    [ "$PREP" = true ] && mpirun -np $np python3 02_prep_pywrdrb_inputs.py "$dataset"
    [ "$SIMULATE" = true ] && mpirun -np $np python3 03_run_pywrdrb_simulations.py "$dataset"
    
echo "Completed generation->simulation for: $dataset"
done    

echo ""
echo "========================================"
echo "All climate scenarios completed successfully!"
echo "========================================"
