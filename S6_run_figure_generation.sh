#!/bin/bash
#SBATCH --job-name=Figs
#SBATCH --output=./logs/figs.out
#SBATCH --error=./logs/fig.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1


# Load modules and environment
module load python/3.11.5
source venv/bin/activate
np=$(($SLURM_NTASKS_PER_NODE * $SLURM_NNODES))

# make directories
mkdir -p logs figures

# Workflow flags
PLOT_ENSEMBLE_DIAGNOSTICS=${PLOT_ENSEMBLE_DIAGNOSTICS:-true}
PLOT_DROUGHT_DISTRIBUTION=${PLOT_DROUGHT_DISTRIBUTION:-true}
PLOT_CONTRIBUTION_KDE=${PLOT_CONTRIBUTION_KDE:-true}
PLOT_CONTRIBUTION_TIMESERIES=${PLOT_CONTRIBUTION_TIMESERIES:-true}
PLOT_PERFORMANCE_OUTCOMES=${PLOT_PERFORMANCE_OUTCOMES:-true}

# Under development still...
PLOT_SANKEY_PARALLEL=${PLOT_SANKEY_PARALLEL:-true}


# Ensemble flow distribution and verification plots
if [ "$PLOT_ENSEMBLE_DIAGNOSTICS" = true ]; then
    echo "========================================"
    echo "Generating ensemble diagnostics figures..."
    echo "========================================"
    python3 F1_plot_ensemble_diagnostics.py stationary_ensemble
fi


if [ "$PLOT_DROUGHT_DISTRIBUTION" = true ]; then
    echo "========================================"
    echo "Generating drought metric distribution figures..."
    echo "========================================"
    python3 F2_plot_drought_metric_distribution.py 12
    python3 F2_plot_drought_metric_distribution.py 6
    python3 F2_plot_drought_metric_distribution.py 3
fi


if [ "$PLOT_CONTRIBUTION_KDE" = true ]; then
    echo "========================================"
    echo "Generating contribution analysis figures..."
    echo "========================================"
    python3 F3_plot_drought_contribution_composite.py
fi


if [ "$PLOT_CONTRIBUTION_TIMESERIES" = true ]; then
    echo "========================================"
    echo "Generating contribution distribution figures..."
    echo "========================================"
    python3 F4_plot_contribution_distributions.py --multipanel
fi


if [ "$PLOT_PERFORMANCE_OUTCOMES" = true ]; then
    echo "========================================"
    echo "Generating performance bar figures..."
    echo "========================================"
    python3 F5_plot_performance_outcomes.py
fi




if [ "$PLOT_SANKEY_PARALLEL" = true ]; then
    echo "========================================"
    echo "Running CART diagnostics for bin selection..."
    echo "========================================"
    # python3 diagnostics_cart_bin_selection.py --ssi_window 3 --datasets stationary_ensemble

    echo "========================================"
    echo "Generating Sankey-Parallel Coordinate figures..."
    echo "========================================"
    python3 F6_plot_sankey_parallel_coordinates.py --versions default quantile cart --ssi_window 3 --datasets stationary_ensemble
fi
