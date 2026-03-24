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
PLOT_ENSEMBLE_DIAGNOSTICS=${PLOT_ENSEMBLE_DIAGNOSTICS:-false}
PLOT_DROUGHT_DISTRIBUTION=${PLOT_DROUGHT_DISTRIBUTION:-false}
PLOT_ZONE_OCCURRENCE=${PLOT_ZONE_OCCURRENCE:-false}
PLOT_CONTRIBUTION_TIMESERIES=${PLOT_CONTRIBUTION_TIMESERIES:-true}
PLOT_PERFORMANCE_OUTCOMES=${PLOT_PERFORMANCE_OUTCOMES:-false}


# F1: Ensemble flow distribution and verification plots
if [ "$PLOT_ENSEMBLE_DIAGNOSTICS" = true ]; then
    echo "========================================"
    echo "F1: Generating ensemble diagnostics figures..."
    echo "========================================"
    python3 F1_plot_ensemble_diagnostics.py stationary_ensemble
fi

# F2: Drought metric distributions
if [ "$PLOT_DROUGHT_DISTRIBUTION" = true ]; then
    echo "========================================"
    echo "F2: Generating drought metric distribution figures..."
    echo "========================================"
    python3 F2_plot_drought_metric_distribution.py 12
    python3 F2_plot_drought_metric_distribution.py 6
    python3 F2_plot_drought_metric_distribution.py 3
fi

# F3: Drought zone occurrence (temporal probability + frequency/duration boxplots)
if [ "$PLOT_ZONE_OCCURRENCE" = true ]; then
    echo "========================================"
    echo "F3: Generating drought zone occurrence figure..."
    echo "========================================"
    python3 F3_plot_drought_zone_occurrence.py
fi

# F4: NYC contribution timeseries
if [ "$PLOT_CONTRIBUTION_TIMESERIES" = true ]; then
    echo "========================================"
    echo "F4: Generating contribution distribution figures..."
    echo "========================================"
    # python3 F4_plot_contribution_distributions.py --combined
    # python3 F4_plot_contribution_distributions.py --montague
    python3 F4_plot_contribution_distributions.py --montague --layout side_by_side
fi

# F5: Performance outcome comparison
if [ "$PLOT_PERFORMANCE_OUTCOMES" = true ]; then
    echo "========================================"
    echo "F5: Generating performance bar figures..."
    echo "========================================"
    python3 F5_plot_performance_outcomes.py
fi

