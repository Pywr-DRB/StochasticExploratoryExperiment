#!/bin/bash
#SBATCH --job-name=Figs
#SBATCH --output=./logs/figs.out
#SBATCH --error=./logs/fig.err
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
PLOT_ENSEMBLE_DIAGNOSTICS=${PLOT_ENSEMBLE_DIAGNOSTICS:-false}
PLOT_NYC_STORAGE_ZONES=${PLOT_NYC_STORAGE_ZONES:-false}
PLOT_DROUGHT_DISTRIBUTION=${PLOT_DROUGHT_DISTRIBUTION:-false}
PLOT_SI_FIGS=${PLOT_SI_FIGS:-false}


# python3 SI7_plot_nyc_contribution_timeseries.py stationary_ensemble
# python3 SI7_plot_nyc_contribution_timeseries.py climate_adjusted_low
# python3 SI7_plot_nyc_contribution_timeseries.py climate_adjusted_high

python3 F4_plot_water_balance_by_drought_zone.py stationary_ensemble
python3 F4_plot_water_balance_by_drought_zone.py climate_adjusted_low
python3 F4_plot_water_balance_by_drought_zone.py climate_adjusted_high


# Ensemble flow distribution and verification plots
if [ "$PLOT_ENSEMBLE_DIAGNOSTICS" = true ]; then
    python3 09_plot_ensemble_diagnostics.py stationary_ensemble
fi


if [ "$PLOT_NYC_STORAGE_ZONES" = true ]; then
    # NYC reservoir storage zone probability plots
    # 4-panel storage zone probability comparison
    python3 09_plot_reservoir_storage_zone_probabilities.py comparison
fi
if [ "$PLOT_DROUGHT_DISTRIBUTION" = true ]; then
    # Drought metric distribution plots

    python3 09_plot_drought_metric_distribution.py stationary_ensemble 3 severity magnitude
    python3 09_plot_drought_metric_distribution.py stationary_ensemble 6 severity magnitude
    python3 09_plot_drought_metric_distribution.py stationary_ensemble 12 severity magnitude

    python3 09_plot_drought_metric_distribution.py climate_adjusted_low 3 severity magnitude
    python3 09_plot_drought_metric_distribution.py climate_adjusted_low 6 severity magnitude
    python3 09_plot_drought_metric_distribution.py climate_adjusted_low 12 severity magnitude

    python3 09_plot_drought_metric_distribution.py climate_adjusted_high 3 severity magnitude
    python3 09_plot_drought_metric_distribution.py climate_adjusted_high 6 severity magnitude
    python3 09_plot_drought_metric_distribution.py climate_adjusted_high 12 severity magnitude
fi


if [ "$PLOT_SI_FIGS" = true ]; then
    # Supplementary Information figures
    python3 SI1_plot_shortage_occurrence_by_day.py stationary_ensemble
    python3 SI2_plot_satisficing_by_event.py stationary_ensemble 12
    python3 SI2_plot_satisficing_by_event.py stationary_ensemble 6
    python3 SI2_plot_satisficing_by_event.py stationary_ensemble 3

    python3 SI1_plot_shortage_occurrence_by_day.py climate_adjusted_low

    python3 SI2_plot_satisficing_by_event.py climate_adjusted_low 12
    python3 SI2_plot_satisficing_by_event.py climate_adjusted_low 6
    python3 SI2_plot_satisficing_by_event.py climate_adjusted_low 3

    python3 SI1_plot_shortage_occurrence_by_day.py climate_adjusted_high
    python3 SI2_plot_satisficing_by_event.py climate_adjusted_high 12
    python3 SI2_plot_satisficing_by_event.py climate_adjusted_high 6
    python3 SI2_plot_satisficing_by_event.py climate_adjusted_high 3
fi


# # 4-panel drought return period comparison
# python3 09_plot_drought_frequency.py comparison

# python3 10_plot_drought_storage_analysis.py climate_adjusted_low 3
# python3 10_plot_drought_storage_analysis.py climate_adjusted_low 6
# python3 10_plot_drought_storage_analysis.py climate_adjusted_low 12





### Drought metric distribution plots
# python3 09_plot_drought_metric_comparison.py 3 severity magnitude baseline climate_adjusted_low
# python3 09_plot_drought_metric_comparison.py 6 severity magnitude baseline climate_adjusted_low
# python3 09_plot_drought_metric_comparison.py 12 severity magnitude baseline climate_adjusted_low


# python3 09_plot_drought_metric_comparison.py 3 severity magnitude baseline climate_adjusted_high
# python3 09_plot_drought_metric_comparison.py 6 severity magnitude baseline climate_adjusted_high
# python3 09_plot_drought_metric_comparison.py 12 severity magnitude baseline climate_adjusted_high


# # 4-panel performance outcome comparison
# python3 09_plot_performance_outcome_bars.py

# python3 09_plot_satisficing_scatter.py --all


