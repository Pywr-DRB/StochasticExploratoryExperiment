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
PLOT_DROUGHT_DISTRIBUTION=${PLOT_DROUGHT_DISTRIBUTION:-false}
PLOT_CONTRIBUTION_ANALYSIS=${PLOT_CONTRIBUTION_ANALYSIS:-true}

# PLOT_NYC_STORAGE_ZONES=${PLOT_NYC_STORAGE_ZONES:-false}


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


if [ "$PLOT_CONTRIBUTION_ANALYSIS" = true ]; then
    echo "========================================"
    echo "Generating contribution analysis figures..."
    echo "========================================"
    # python3 F3_plot_drought_contribution_composite.py

    echo "========================================"
    echo "Generating contribution distribution figures..."
    echo "========================================"
    python3 F5_plot_contribution_distributions.py --multipanel
fi



