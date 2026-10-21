#!/bin/bash
#SBATCH --job-name=SI
#SBATCH --output=./logs/SI.out
#SBATCH --error=./logs/SI.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --time=48:00:00

# Load modules and environment
module load python/3.11.5
source venv/bin/activate

mkdir -p logs

run() {
    echo "==== Running: $* ===="
    python3 "$@"
    if [ $? -ne 0 ]; then
        echo "ERROR: $* failed"
        exit 1
    fi
}

### SI0: Ensemble diagnostics at every generation node (SI Figures S2.6 to S2.8)
run si_scripts/SI0_full_ensemble_diagnostics.py stationary_ensemble

### SI0b: Low-flow ensemble convergence (SI Figure S2.9)
run si_scripts/SI0b_low_flow_convergence.py stationary_ensemble

### SI2: Diversion demand diagnostics (SI Figures S2.2 to S2.4)
run si_scripts/SI2_plot_diversion_diagnostics.py stationary_ensemble
run si_scripts/SI2_plot_diversion_diagnostics.py climate_adjusted_low
run si_scripts/SI2_plot_diversion_diagnostics.py climate_adjusted_high

### SI3: storage zone probabilities are calculated in S5_calculate_performance_metrics.sh

### SI15: Share of Drought Emergency events captured by each SSI window (SI Figure S3.1)
run si_scripts/SI15_plot_ssi_window_emergency_capture.py

### SI18: Climate-adjusted ensemble monthly-flow validation (SI Figure S2.1)
run si_scripts/SI18_climate_adjustment_validation.py

### SI19: Convergence of the joint-exceedance drought return period (SI Figures S2.10 to S2.12)
run si_scripts/SI19_drought_return_period_convergence.py stationary_ensemble 3
run si_scripts/SI19_drought_return_period_convergence.py climate_adjusted_low 3
run si_scripts/SI19_drought_return_period_convergence.py climate_adjusted_high 3

### SI20: NYC aggregate inflow duration-severity curves (SI Figure S3.4)
run si_scripts/SI20_duration_severity_curves.py

### SI22: Storyline droughts (Events A and B) selected from each ensemble (SI Figures S6.1 to S6.3)
run si_scripts/SI22_scenario_storyline_droughts.py

echo ""
echo "========================================"
echo "All SI scripts completed successfully."
echo "========================================"
