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

# Configuration name (determines output directory)
export CONFIG_NAME=${CONFIG_NAME:-default}

mkdir -p logs

run() {
    echo "==== Running: $* ===="
    python3 "$@"
    if [ $? -ne 0 ]; then
        echo "ERROR: $* failed"
        exit 1
    fi
}

### SI0: Detailed stationary ensemble diagnostics
run si_scripts/SI0_full_ensemble_diagnostics.py stationary_ensemble

### SI1: NYC, Montague, Trenton shortage occurrence by day of year
run si_scripts/SI1_plot_shortage_occurrence_by_day.py stationary_ensemble

### SI2: Diversion diagnostics/distributions
run si_scripts/SI2_plot_diversion_diagnostics.py stationary_ensemble
run si_scripts/SI2_plot_diversion_diagnostics.py climate_adjusted_low
run si_scripts/SI2_plot_diversion_diagnostics.py climate_adjusted_high

### SI3 + SI4: Storage zone probabilities (calculate and plot)
run si_scripts/SI3_calculate_storage_zone_probabilities.py --all
run si_scripts/SI4_plot_reservoir_storage_zone_probabilities.py comparison

### SI5: NYC diversions relative to storage zone classification
run si_scripts/SI5_plot_shortages_by_zone.py stationary_ensemble

### SI6: Contribution ratio for years with storage < 20%
run si_scripts/SI6_plot_contribution_storage_timeseries.py --multipanel

### SI7: Drought event scatter colored by satisficing outcomes
run si_scripts/SI7_plot_drought_satisficing_scatter.py 3
run si_scripts/SI7_plot_drought_satisficing_scatter.py 6
run si_scripts/SI7_plot_drought_satisficing_scatter.py 12

### SI8: Rank contributions and storage for percentile-based realizations
run si_scripts/SI8_plot_storage_montague_exceedance.py 0.5
run si_scripts/SI8_plot_storage_montague_exceedance.py 0.1

### SI9: All performance metrics as CDFs
run si_scripts/SI9_plot_metric_distributions.py

### SI10: Montague/Trenton shortage by NYC storage zone
run si_scripts/SI10_plot_montague_trenton_shortage_by_zone.py stationary_ensemble

### SI11: Lower basin reservoir storage
run si_scripts/SI11_plot_lower_basin_reservoir_storage.py

### SI12: Drought duration vs magnitude scatter
run si_scripts/SI12_duration_magnitude_scatter.py

### SI13: Drought outcome heatmaps (severity x magnitude)
run si_scripts/SI13_plot_drought_heatmap_with_storage_outcomes.py 3

### SI14: Drought magnitude vs minimum storage
run si_scripts/SI14_plot_drought_storage_3d_scatter.py 3

### SI15: SSI window drought emergency capture rate
run si_scripts/SI15_plot_ssi_window_emergency_capture.py

### SI16: Performance outcomes (exceedance curves, ridgelines, quantile strips)
run si_scripts/SI16_plot_performance_outcomes.py

echo ""
echo "========================================"
echo "All SI scripts completed successfully."
echo "========================================"
