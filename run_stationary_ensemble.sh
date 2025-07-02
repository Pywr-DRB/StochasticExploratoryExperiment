#!/bin/bash
#SBATCH --job-name=SE-DRB
#SBATCH --output=./logs/exploratory.out
#SBATCH --error=./logs/exploratory.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --exclusive


module load python/3.11.5
source venv/bin/activate
np=$(($SLURM_NTASKS_PER_NODE * $SLURM_NNODES))

### Run baseline simulations on single task
# Runs simulations using all existing pywrdrb datasets
echo "Running baseline simulations..."
time mpirun -np $np python3 00_run_baseline_simulations.py


# ### Generate stationary ensemble
# # Use Kirsch-Nowak to generate ensemble using reconstruction
# # as the source data
# echo "Generating stationary ensemble..."
# python 01_generate_stationary_ensemble.py


# ### Plot ensemble diagnostics
# # Make statistical diagnostic figures 
# echo "Plotting ensemble diagnostics..."
# python 02_plot_ensemble_diagnostics.py