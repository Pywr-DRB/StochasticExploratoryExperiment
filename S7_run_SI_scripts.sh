#!/bin/bash
#SBATCH --job-name=SI
#SBATCH --output=./logs/SI.out
#SBATCH --error=./logs/SI.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00


# Load modules and environment
module load python/3.11.5
source venv/bin/activate
np=$(($SLURM_NTASKS_PER_NODE * $SLURM_NNODES))

# make directories
mkdir -p logs figures

# Workflow flags
PLOT_ENSEMBLE_DIAGNOSTICS=${PLOT_ENSEMBLE_DIAGNOSTICS:-true}
PLOT_SHORTAGE_OCCURRENCE=${PLOT_SHORTAGE_OCCURRENCE:-true}
PLOT_DIVERSION_DIAGNOSTICS=${PLOT_DIVERSION_DIAGNOSTICS:-true}
PLOT_STORAGE_ZONE_PROBABILITIES=${PLOT_STORAGE_ZONE_PROBABILITIES:-true}
PLOT_NYC_DIVERSION_BY_ZONE=${PLOT_NYC_DIVERSION_BY_ZONE:-true}
PLOT_CONTRIBUTION_TIMESERIES=${PLOT_CONTRIBUTION_TIMESERIES:-true}
PLOT_DROUGHT_SATISFICING=${PLOT_DROUGHT_SATISFICING:-true}
PLOT_STORAGE_EXCEEDANCE=${PLOT_STORAGE_EXCEEDANCE:-true}
PLOT_METRIC_DISTRIBUTIONS=${PLOT_METRIC_DISTRIBUTIONS:-true}
PLOT_SHORTAGE_BY_ZONE=${PLOT_SHORTAGE_BY_ZONE:-true}
PLOT_LOWER_BASIN_STORAGE=${PLOT_LOWER_BASIN_STORAGE:-true}


### SI0: Detailed stationary ensemble diagnostics
if [ "$PLOT_ENSEMBLE_DIAGNOSTICS" = true ]; then
    echo "========================================"
    echo "SI0: Generating ensemble diagnostics..."
    echo "========================================"
    python3 SI0_full_ensemble_diagnostics.py stationary_ensemble
fi


### SI1: NYC, Montague, Trenton shortage occurrence by day of year
if [ "$PLOT_SHORTAGE_OCCURRENCE" = true ]; then
    echo "========================================"
    echo "SI1: Generating shortage occurrence by day figures..."
    echo "========================================"
    python3 SI1_plot_shortage_occurrence_by_day.py stationary_ensemble
fi


### SI2: Diversion diagnostics/distributions
if [ "$PLOT_DIVERSION_DIAGNOSTICS" = true ]; then
    echo "========================================"
    echo "SI2: Generating diversion diagnostics figures..."
    echo "========================================"
    python3 SI2_plot_diversion_diagnostics.py stationary_ensemble
    python3 SI2_plot_diversion_diagnostics.py climate_adjusted_low
    python3 SI2_plot_diversion_diagnostics.py climate_adjusted_high
fi


### SI3 + SI4: Storage zone probabilities (calculate and plot)
if [ "$PLOT_STORAGE_ZONE_PROBABILITIES" = true ]; then
    echo "========================================"
    echo "SI3-SI4: Calculating and plotting storage zone probabilities..."
    echo "========================================"
    python3 SI3_calculate_storage_zone_probabilities.py --all
    python3 SI4_plot_reservoir_storage_zone_probabilities.py comparison
fi


### SI5: NYC diversions relative to storage zone classification
if [ "$PLOT_NYC_DIVERSION_BY_ZONE" = true ]; then
    echo "========================================"
    echo "SI5: Generating NYC diversion shortage by zone figures..."
    echo "========================================"
    python3 SI5_plot_nyc_diversion_shortage_by_zone.py stationary_ensemble
fi


### SI6: Contribution ratio for years with storage < 20%
if [ "$PLOT_CONTRIBUTION_TIMESERIES" = true ]; then
    echo "========================================"
    echo "SI6: Generating contribution storage timeseries figures..."
    echo "========================================"
    python3 SI6_plot_contribution_storage_timeseries.py --multipanel
fi


### SI7: Drought event scatter colored by satisficing outcomes
if [ "$PLOT_DROUGHT_SATISFICING" = true ]; then
    echo "========================================"
    echo "SI7: Generating drought satisficing scatter figures..."
    echo "========================================"
    python3 SI7_plot_drought_satisficing_scatter.py 3
    python3 SI7_plot_drought_satisficing_scatter.py 6
    python3 SI7_plot_drought_satisficing_scatter.py 12
fi


### SI8: Rank contributions and storage for percentile-based realizations
if [ "$PLOT_STORAGE_EXCEEDANCE" = true ]; then
    echo "========================================"
    echo "SI8: Generating storage/Montague exceedance figures..."
    echo "========================================"
    python3 SI8_plot_storage_montague_exceedance.py 0.5
    python3 SI8_plot_storage_montague_exceedance.py 0.1
fi


### SI9: All performance metrics as CDFs
if [ "$PLOT_METRIC_DISTRIBUTIONS" = true ]; then
    echo "========================================"
    echo "SI9: Generating metric distribution figures..."
    echo "========================================"
    python3 SI9_plot_metric_distributions.py
fi


### SI10: Montague/Trenton shortage by NYC storage zone
if [ "$PLOT_SHORTAGE_BY_ZONE" = true ]; then
    echo "========================================"
    echo "SI10: Generating Montague/Trenton shortage by zone figures..."
    echo "========================================"
    python3 SI10_plot_montague_trenton_shortage_by_zone.py stationary_ensemble
fi


### SI11: Lower basin reservoir storage
if [ "$PLOT_LOWER_BASIN_STORAGE" = true ]; then
    echo "========================================"
    echo "SI11: Generating lower basin reservoir storage figures..."
    echo "========================================"
    python3 SI11_plot_lower_basin_reservoir_storage.py
fi
