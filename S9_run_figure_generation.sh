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

# Workflow flags
PLOT_ENSEMBLE_DIAGNOSTICS=${PLOT_ENSEMBLE_DIAGNOSTICS:-true}
PLOT_NYC_STORAGE_ZONES=${PLOT_NYC_STORAGE_ZONES:-false}

# Ensemble flow distribution and verification plots
if [ "$PLOT_ENSEMBLE_DIAGNOSTICS" = true ]; then
    python3 09_plot_ensemble_diagnostics.py stationary_ensemble
fi


if [ "$PLOT_NYC_STORAGE_ZONES" = true ]; then
    # NYC reservoir storage zone probability plots
    # 4-panel storage zone probability comparison
    python3 09_plot_reservoir_storage_zone_probabilities.py comparison
fi




# # 4-panel drought return period comparison
# python3 09_plot_drought_frequency.py comparison

# python3 10_plot_drought_storage_analysis.py climate_adjusted_low 3
# python3 10_plot_drought_storage_analysis.py climate_adjusted_low 6
# python3 10_plot_drought_storage_analysis.py climate_adjusted_low 12





### Drought metric distribution plots

# python3 09_plot_drought_metric_distribution.py stationary_ensemble 3 severity magnitude
# python3 09_plot_drought_metric_distribution.py stationary_ensemble 6 severity magnitude
# python3 09_plot_drought_metric_distribution.py stationary_ensemble 12 severity magnitude


# python3 09_plot_drought_metric_distribution.py climate_adjusted_low 3 severity magnitude
# python3 09_plot_drought_metric_distribution.py climate_adjusted_low 6 severity magnitude
# python3 09_plot_drought_metric_distribution.py climate_adjusted_low 12 severity magnitude

# python3 09_plot_drought_metric_distribution.py climate_adjusted_high 3 severity magnitude
# python3 09_plot_drought_metric_distribution.py climate_adjusted_high 6 severity magnitude
# python3 09_plot_drought_metric_distribution.py climate_adjusted_high 12 severity magnitude


# python3 09_plot_drought_metric_comparison.py 3 severity magnitude baseline climate_adjusted_low
# python3 09_plot_drought_metric_comparison.py 6 severity magnitude baseline climate_adjusted_low
# python3 09_plot_drought_metric_comparison.py 12 severity magnitude baseline climate_adjusted_low


# python3 09_plot_drought_metric_comparison.py 3 severity magnitude baseline climate_adjusted_high
# python3 09_plot_drought_metric_comparison.py 6 severity magnitude baseline climate_adjusted_high
# python3 09_plot_drought_metric_comparison.py 12 severity magnitude baseline climate_adjusted_high


# # 4-panel performance outcome comparison
# python3 09_plot_performance_outcome_bars.py

# python3 09_plot_satisficing_scatter.py --all


