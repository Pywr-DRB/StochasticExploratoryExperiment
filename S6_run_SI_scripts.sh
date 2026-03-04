#!/bin/bash
#SBATCH --job-name=SI
#SBATCH --output=./logs/SI.out
#SBATCH --error=./logs/SI.err
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

### Detailed stationary ensemble diagnostics to confirm generator
python3 SI0_full_ensemble_diagnostics.py stationary_ensemble


### NYC, Montague, Trenton Shortage occurence by day of the year 
python3 SI1_plot_shortage_occurrence_by_day.py stationary_ensemble


### Diversion diagnostics/distributions
python3 SI2_plot_diversion_diagnostics.py stationary_ensemble
python3 SI2_plot_diversion_diagnostics.py climate_adjusted_low
python3 SI2_plot_diversion_diagnostics.py climate_adjusted_high


### Calculate and plot storage zone probabilites
python3 SI3_calculate_storage_zone_probabilities.py --all
python3 SI4_plot_reservoir_storage_zone_probabilities.py comparison

### NYC diversions relative to storage zone classification
python3 SI5_plot_nyc_diversion_shortage_by_zone.py stationary_ensemble


### contribution ratio for years storage < 20
python3 SI6_plot_contribution_storage_timeseries.py --multipanel

### Drought event scatter colored by satisficing outcomes
python3 SI7_plot_drought_satisficing_scatter.py 3
python3 SI7_plot_drought_satisficing_scatter.py 6
python3 SI7_plot_drought_satisficing_scatter.py 12

### Rank contributions and storage for percentile-based realizations 
python3 SI8_plot_storage_montague_exceedance.py 0.5
python3 SI8_plot_storage_montague_exceedance.py 0.1




### UNFINISHED #############
