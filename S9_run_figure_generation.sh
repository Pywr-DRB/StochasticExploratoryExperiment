#!/bin/bash
#SBATCH --job-name=Figs
#SBATCH --output=./logs/fig_generation.out
#SBATCH --error=./logs/fig_generation.err
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

# Ensemble diagnostic plots for delMontague
# python3 10_plot_streamflow_scenario_comparison.py delMontague

# # 4-panel drought return period comparison
# python3 09_plot_drought_frequency.py comparison

# # 4-panel storage zone probability comparison
# python3 09_plot_reservoir_storage_zone_probabilities.py comparison

# # 4-panel performance outcome comparison
python3 09_plot_performance_outcome_bars.py

# python3 09_plot_satisficing_scatter.py --all




### OLD
# python3 09_plot_drought_frequency.py "stationary_ensemble"
# python3 09_plot_drought_frequency.py "climate_adjusted_ssp245_min"
# python3 09_plot_drought_frequency.py "climate_adjusted_ssp245_max"
# python3 09_plot_drought_frequency.py "climate_adjusted_ssp245_median"
# python3 09_plot_drought_frequency.py "climate_adjusted_ssp370_min"
# python3 09_plot_drought_frequency.py "climate_adjusted_ssp370_max"
# python3 09_plot_drought_frequency.py "climate_adjusted_ssp370_median"
