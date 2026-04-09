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

# # Fig4: Ensemble flow distribution and verification plots
# run plotting_scripts/Fig4_plot_ensemble_diagnostics.py stationary_ensemble

# # Fig5: Drought metric distributions
# # run plotting_scripts/Fig5_plot_drought_metric_distribution.py 12
# # run plotting_scripts/Fig5_plot_drought_metric_distribution.py 6
# run plotting_scripts/Fig5_plot_drought_metric_distribution.py 3

# # Fig6: Drought zone occurrence (temporal probability + frequency/duration boxplots)
# run plotting_scripts/Fig6_plot_drought_zone_occurrence.py


# # Fig7: KDEs of NYC contribution / total inflow
# run plotting_scripts/Fig7_nyc_contribution_kdes.py

# # Fig8: NYC contribution distributions
# run plotting_scripts/Fig8_plot_contribution_distributions.py --montague --layout side_by_side

# Fig9: Drought satisficing heatmaps (severity x magnitude)
run plotting_scripts/Fig9_plot_drought_satisficing_heatmap.py 3

# Fig10: State dynamics for selected drought bin
run plotting_scripts/Fig10_drought_dynamics.py

echo ""
echo "========================================"
echo "All figure scripts completed successfully."
echo "========================================"
