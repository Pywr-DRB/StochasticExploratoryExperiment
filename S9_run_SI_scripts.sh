#!/bin/bash
#SBATCH --job-name=SI
#SBATCH --output=./logs/SI.out
#SBATCH --error=./logs/SI.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=0


# Load modules and environment
module load python/3.11.5
source venv/bin/activate
np=$(($SLURM_NTASKS_PER_NODE * $SLURM_NNODES))

# make directories
mkdir -p logs figures


python3 SI1_plot_shortage_occurrence_by_day.py stationary_ensemble

python3 SI2_plot_satisficing_by_event.py stationary_ensemble 3
python3 SI2_plot_satisficing_by_event.py stationary_ensemble 6
python3 SI2_plot_satisficing_by_event.py stationary_ensemble 12