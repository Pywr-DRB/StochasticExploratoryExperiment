#!/bin/bash
#SBATCH --job-name=BKN
#SBATCH --output=./logs/Borg.out
#SBATCH --error=./logs/Borg.err
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=40
#SBATCH --exclusive

# Load Python module
module load python/3.11.5

# Activate Python virtual environment
source venv/bin/activate

# Define function to submit a single job iteration


datetime=$(date '+%Y-%m-%d %H:%M:%S')
n_processors=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

echo "Datetime: $datetime"
echo "Total processors: $n_processors"

# Run with MPI
time mpirun --oversubscribe -np $n_processors python kirsch_borg_run.py

echo "Generating synthetic ensemble from Borg output solutions..."
time mpirun --oversubscribe -np $n_processors python kirsch_borg_generate.py

echo "Plotting ensemble diagnostics..."
time python 02_plot_ensemble_diagnostics.py

echo "Completed!"