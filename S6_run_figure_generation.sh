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
PLOT_DROUGHT_DISTRIBUTION=${PLOT_DROUGHT_DISTRIBUTION:-false}
PLOT_ZONE_OCCURRENCE=${PLOT_ZONE_OCCURRENCE:-false}
PLOT_CONTRIBUTION_DISTRIBUTIONS=${PLOT_CONTRIBUTION_DISTRIBUTIONS:-false}
PLOT_SATISFICING_HEATMAP=${PLOT_SATISFICING_HEATMAP:-false}


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

# F4: NYC contribution distributions
if [ "$PLOT_CONTRIBUTION_DISTRIBUTIONS" = true ]; then
    echo "========================================"
    echo "F4: Generating contribution distribution figures..."
    echo "========================================"
    python3 F4_plot_contribution_distributions.py --montague --layout side_by_side
fi

# F5: Drought satisficing heatmaps (severity × magnitude)
if [ "$PLOT_SATISFICING_HEATMAP" = true ]; then
    echo "========================================"
    echo "F5: Generating drought satisficing heatmap figures..."
    echo "========================================"
    python3 F5_plot_drought_satisficing_heatmap.py 3
fi
