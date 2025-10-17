#!/bin/bash
#SBATCH --job-name=Figs
#SBATCH --output=./logs/fig_generation.out
#SBATCH --error=./logs/fig_generation.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=0
#SBATCH --exclusive

# Load modules and environment
module load python/3.11.5
source venv/bin/activate
np=$(($SLURM_NTASKS_PER_NODE * $SLURM_NNODES))

# make directories
mkdir -p logs figures

# python3 09_plot_reservoir_storage_zone_probabilities.py --all

python3 09_plot_performance_outcome_bars.py

# python3 09_plot_drought_frequency.py "stationary_ensemble"
# python3 09_plot_drought_frequency.py "climate_adjusted_ssp245_min"
# python3 09_plot_drought_frequency.py "climate_adjusted_ssp245_max"
# python3 09_plot_drought_frequency.py "climate_adjusted_ssp245_median"
# python3 09_plot_drought_frequency.py "climate_adjusted_ssp370_min"
# python3 09_plot_drought_frequency.py "climate_adjusted_ssp370_max"
# python3 09_plot_drought_frequency.py "climate_adjusted_ssp370_median"
