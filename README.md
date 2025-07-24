# StochasticExploratoryExperiment

**Work in Progress: Exploratory modeling in the Delaware River Basin (DRB)**

## Overview

This repository implements a high-performance stochastic ensemble modeling framework for the Delaware River Basin using the Pywr-DRB water resources simulation platform. The workflow generates synthetic streamflow ensembles and evaluates water system performance under both stationary and climate-adjusted conditions.

## Project Architecture

### Core Components

1. **Synthetic Flow Generation**: Uses Kirsch-Nowak (KN) methodology to generate synthetic streamflow ensembles
2. **Parallel Processing**: MPI-based distributed computing for large-scale ensemble generation and simulation
3. **Water System Simulation**: Pywr-DRB integration for reservoir operations and water allocation modeling
4. **Performance Analysis**: Drought metrics, reliability analysis, and system performance evaluation

### Ensemble Configuration

- **Total Realizations**: 1,000 synthetic streamflow realizations
- **Ensemble Sets**: 10 sets of 100 realizations each (memory-optimized processing)
- **Simulation Period**: 70 years (1950-2019)
- **Temporal Resolution**: Daily flows with monthly analysis
- **Spatial Coverage**: Major DRB nodes and tributaries

## Workflow Structure

### Step 1: Ensemble Generation
```
01_generate_stationary_ensemble_sets.py
01_generate_climate_adjusted_ensemble_sets.py
```
- Parallel generation of synthetic streamflow ensembles using MPI
- Stationary ensembles preserve historical flow statistics
- Climate-adjusted ensembles apply monthly mean flow shifts
- Automatic MPI rank distribution across ensemble sets

### Step 2: Simulation Preprocessing
```
02_prep_pywrdrb_inputs.py
```
- Converts synthetic flows to Pywr-DRB compatible format
- Applies spatial and temporal disaggregation
- Generates predicted inflow files for simulation

### Step 3: Water System Simulation
```
03_run_pywrdrb_simulations.py
```
- Distributed Pywr-DRB simulations across ensemble sets
- Batch processing within sets for memory management
- Outputs reservoir operations, diversions, and flow targets

### Step 4: Analysis and Visualization
```
04_plot_ensemble_diagnostics.py
05_calculate_ssi_drought_metrics.py
06_calculate_hashimoto_metrics_during_droughts.py
```
- Statistical validation of synthetic flows
- Drought characterization using Standardized Streamflow Index (SSI)
- Reliability-resilience-vulnerability analysis
- Flow duration curves and spatial correlation analysis

## Key Features

### Ensemble Types
- **Stationary**: Preserves historical flow statistics and variability
- **Climate-Adjusted**: Applies prescribed monthly mean flow changes


## Configuration

### System Requirements
- Python 3.11+
- MPI implementation (tested with OpenMPI)
- HPC nodes (recommended: 5 nodes × 40 cores)

### Key Parameters (`config.py`)
```python
TOTAL_REALIZATIONS = 1000
N_REALIZATIONS_PER_ENSEMBLE_SET = 100
N_ENSEMBLE_SETS = 10
N_YEARS = 70
N_REALIZATIONS_PER_PYWRDRB_BATCH = 10
```


## Usage

### SLURM
```bash
# Stationary ensemble workflow
sbatch S1_run_stationary_ensemble.sh

# Climate-adjusted ensemble workflow  
sbatch S2_run_climate_adjusted_ensemble.sh

# Post-processing and analysis
sbatch S9_run_postprocessing.sh
```

### Development Environment
```bash
module load python/3.11.5
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## File Structure

```
├── config.py                    # Main configuration parameters
├── methods/
│   ├── load.py                 # Data loading utilities
│   ├── metrics/                # Performance metrics
│   ├── plotting/               # Visualization tools
│   └── utils.py                # Helper functions
├── pywrdrb/
│   ├── inputs/                 # Generated ensemble data
│   ├── outputs/                # Simulation results
│   └── drought_metrics/        # Drought analysis results
├── figures/                    # Generated plots and diagnostics
└── logs/                       # Execution logs
```

## Outputs

### Ensemble Data
- Synthetic gage flows (HDF5 format)
- Catchment inflows for simulation
- Predicted inflow files for Pywr-DRB

### Simulation Results
- Reservoir storage timeseries
- Flow target violations
- Inter-basin transfer performance
- System reliability metrics

### Analysis Products
- Flow duration curve comparisons
- Drought event catalogs
- Reliability-resilience-vulnerability metrics
- Spatial correlation validation

## Dependencies

- `pywrdrb`: Water resources simulation platform
- `sglib`: Stochastic generation and analysis tools
- `mpi4py`: MPI parallelization
- `pandas`, `numpy`: Data manipulation
- `matplotlib`, `seaborn`: Visualization
