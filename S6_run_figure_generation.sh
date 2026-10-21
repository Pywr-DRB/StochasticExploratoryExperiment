#!/bin/bash
#SBATCH --job-name=Figs
#SBATCH --output=./logs/figs.out
#SBATCH --error=./logs/figs.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0

# Load modules and environment
module load python/3.11.5
source venv/bin/activate

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

# Fig4: Ensemble diagnostics (SI Figure S2.5)
run plotting_scripts/Fig4_plot_ensemble_diagnostics.py stationary_ensemble

# Fig5: Drought metric distributions (Figure 4 for SSI-3; SI Figures S3.2 and S3.3 for SSI-6 and SSI-12)
run plotting_scripts/Fig5_plot_drought_metric_distribution.py 12
run plotting_scripts/Fig5_plot_drought_metric_distribution.py 6
run plotting_scripts/Fig5_plot_drought_metric_distribution.py 3

# Fig6: Drought zone seasonality and occurrence (Figure 5)
run plotting_scripts/Fig6_plot_drought_zone_occurrence.py

# Fig7: NYC contribution and diversion KDEs by storage zone (SI Figure S5.1)
run plotting_scripts/Fig7_2x3_kdes_3mo_9mo_windows.py

# Fig8: NYC contribution to the Montague flow target across the water year (Figure 6)
run plotting_scripts/Fig8_plot_contribution_distributions.py --montague --layout stacked

# Fig9: Consequential drought region heatmaps (Figure 7)
run plotting_scripts/Fig9_plot_drought_satisficing_heatmap_2col.py 3

# Fig10: Drought storylines (Figure 8)
run plotting_scripts/Fig10_focal_drought_dynamics.py

echo ""
echo "========================================"
echo "All figure scripts completed successfully."
echo "========================================"
