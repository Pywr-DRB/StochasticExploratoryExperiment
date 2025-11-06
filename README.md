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

## File naming convention

The `archive/` contains old scripts that are no long used and can be ignored. 

- `S*.sh` scripts contain the bash job scripts used to run the full workflow.
- `0*.py` sctipts are the main workflow running generation, simulation and post processing.
- `F*.py` scripts are the final figure generation scripts used in the manuscript. 
- `SI*.py` scripts contain analyses that are used for supporting information, but not part of the main manuscript results. 



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

### To be continued...


***

## Plotting Scripts:
