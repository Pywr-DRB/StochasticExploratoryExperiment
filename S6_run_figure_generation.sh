#!/bin/bash
#SBATCH --job-name=Figs
#SBATCH --output=./logs/figs.out
#SBATCH --error=./logs/fig.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Load modules and environment
module load python/3.11.5
source venv/bin/activate

# Configuration name (determines output directory)
export CONFIG_NAME=${CONFIG_NAME:-default}

mkdir -p logs

run() {
    echo "========================================"
    echo "Running: $*"
    echo "========================================"
    python3 "$@"
    if [ $? -ne 0 ]; then
        echo "ERROR: $* failed"
        exit 1
    fi
}

# F1: Ensemble flow distribution and verification plots
run plotting_scripts/F1_plot_ensemble_diagnostics.py stationary_ensemble

# F2: Drought metric distributions
run plotting_scripts/F2_plot_drought_metric_distribution.py 12
run plotting_scripts/F2_plot_drought_metric_distribution.py 6
run plotting_scripts/F2_plot_drought_metric_distribution.py 3

# F3: Drought zone occurrence (temporal probability + frequency/duration boxplots)
run plotting_scripts/F3_plot_drought_zone_occurrence.py

# F4: NYC contribution distributions
run plotting_scripts/F4_plot_contribution_distributions.py --montague --layout side_by_side

# F5: Drought satisficing heatmaps (severity x magnitude)
run plotting_scripts/F5_plot_drought_satisficing_heatmap.py 3

echo ""
echo "========================================"
echo "All figure scripts completed successfully."
echo "========================================"
