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

# SSI copula diagnostic figures (for Supplementary Information)
python3 07_compare_copula_parameters.py


# SSI copula diagnostic figures (for Supplementary Information)
# python3 09_plot_ssi_copula_diagnostics.py

# Ensemble diagnostic plots for delMontague
# python3 10_plot_streamflow_scenario_comparison.py delMontague

# # 4-panel drought return period comparison
# python3 09_plot_drought_frequency.py comparison

# # 4-panel storage zone probability comparison
# python3 09_plot_reservoir_storage_zone_probabilities.py comparison

# # 4-panel performance outcome comparison
# python3 09_plot_performance_outcome_bars.py

# python3 09_plot_satisficing_scatter.py --all


