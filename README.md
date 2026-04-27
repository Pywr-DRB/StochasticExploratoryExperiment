# Stochastic Exploratory Experiment

Stochastic ensemble modeling framework for the Delaware River Basin (DRB), built on the [Pywr-DRB](https://github.com/Pywr-DRB/Pywr-DRB) water resources simulation platform. The framework generates large synthetic streamflow ensembles using the Kirsch-Nowak methodology, simulates water system operations, and evaluates system performance under stationary and climate-adjusted conditions.

## Dependencies

Listed in `requirements.txt`. Key packages:

- [Pywr-DRB](https://github.com/Pywr-DRB/Pywr-DRB) -- Water resources model for the DRB
- [SynHydro](https://github.com/TrevorJA/SynHydro) -- Stochastic generation library (Kirsch-Nowak, SSI)
- [mpi4py](https://mpi4py.readthedocs.io/) -- MPI-based parallelization

Install with:
```
pip install -r requirements.txt
```

## Workflow

The pipeline is executed through numbered scripts. Each step accepts a `dataset_id` argument (`stationary_ensemble`, `climate_adjusted_low`, or `climate_adjusted_high`) and supports MPI parallelization.

| Script | Description |
|--------|-------------|
| `00_run_baseline_simulations.py` | Run baseline Pywr-DRB model with historical flows |
| `01_generate_ensemble_sets.py` | Generate synthetic streamflow ensembles (Kirsch-Nowak) |
| `02_prep_pywrdrb_inputs.py` | Convert synthetic flows to Pywr-DRB input format |
| `03_run_pywrdrb_simulations.py` | Run Pywr-DRB simulations across ensemble sets |
| `04_postprocess_data_mpi.py` | Postprocess HDF5 simulation outputs (shortage, contribution, zone events) |
| `05_calculate_ssi_drought_metrics.py` | Calculate SSI-based drought metrics (3, 6, 12-month windows) |
| `06_calculate_performance_metrics.py` | Annual performance metrics, Hashimoto RRV, event metrics |

Example usage:
```bash
mpirun -np 150 python 01_generate_ensemble_sets.py stationary_ensemble
```

A serial workflow (`serial_workflow.py`) is available for debugging or small-scale runs without MPI.

## SLURM submission scripts

| Script | Description | Resources |
|--------|-------------|-----------|
| `S0_run_baseline_historic.sh` | Run baseline historical simulations | 1 node |
| `S1_run_stationary_ensemble.sh` | Generate, prep, and simulate stationary ensemble | 8 nodes, 30 tasks/node |
| `S2_run_climate_adjusted_ensemble.sh` | Generate, prep, and simulate climate-adjusted ensembles | 8 nodes, 30 tasks/node |
| `S3_postprocess_all.sh` | Postprocess all 3 datasets sequentially | 1 node, 20 tasks |
| `S4_calculate_ssi.sh` | Calculate SSI drought metrics | 8 nodes, 40 tasks/node |
| `S5_calculate_performance_metrics.sh` | Calculate annual, Hashimoto, and event metrics | 1 node, 20 tasks |
| `S6_run_figure_generation.sh` | Generate manuscript figures | 1 node |
| `S7_run_SI_scripts.sh` | Generate supplementary information figures | 1 node |
| `S8_extract_manuscript_values.sh` | Extract Section 4 manuscript reference values from ensemble outputs | 1 node, 1 task |
| `S99_run_entire_workflow.sh` | Submit full pipeline with SLURM dependency chains | All of the above |

Usage:
```bash
# Run the entire pipeline (jobs are chained with SLURM dependencies):
bash S99_run_entire_workflow.sh

# Or submit individual steps:
sbatch S0_run_baseline_historic.sh
sbatch S1_run_stationary_ensemble.sh
sbatch S2_run_climate_adjusted_ensemble.sh
sbatch S3_postprocess_all.sh
sbatch S4_calculate_ssi.sh
sbatch S5_calculate_performance_metrics.sh
sbatch S6_run_figure_generation.sh
sbatch S7_run_SI_scripts.sh
sbatch S8_extract_manuscript_values.sh
```

`S99_run_entire_workflow.sh` uses `--dependency=afterok` to chain jobs, maximizing parallelism:

```
S0 (baseline)
 |
 ├── S1 (stationary)
 └── S2 (climate-adjusted)
      |
      S3 (postprocess)
      |
      S4 (SSI metrics)
      |
      S5 (performance metrics)
      |
      ├── S6 (figures)
      ├── S7 (SI figures)
      └── S8 (extract manuscript values)
```

## File naming conventions

- `S*.sh` -- SLURM job submission scripts for HPC execution
- `0*.py` -- Main workflow scripts (generation, simulation, post-processing)
- `plotting_scripts/F*.py` -- Manuscript figure generation scripts
- `si_scripts/SI*.py` -- Supplementary information figure scripts

## Configuration

All ensemble and experiment parameters are defined in `methods/config.py`, including:

- Number of realizations and ensemble sets
- Simulation period and temporal resolution
- Dataset definitions (stationary and climate-adjusted scenarios)
- Pywr-DRB batching and output settings

### Output isolation with CONFIG_NAME

All simulation outputs and figures are written to a config-specific directory under `outputs/`. Set the `CONFIG_NAME` environment variable to isolate results from different experiment configurations:

```bash
# Default config
python 03_run_pywrdrb_simulations.py stationary_ensemble
# → outputs/default/data/simulations/...

# Named config
CONFIG_NAME=perfect_foresight python 03_run_pywrdrb_simulations.py stationary_ensemble
# → outputs/perfect_foresight/data/simulations/...

# Set for entire session
export CONFIG_NAME=regression_disagg
python 03_run_pywrdrb_simulations.py stationary_ensemble
# → outputs/regression_disagg/data/...

# SLURM
CONFIG_NAME=regression_disagg sbatch S6_run_figure_generation.sh
```

Each config directory contains a `config.json` recording the settings used.

## Project structure

```
plotting_scripts/     Manuscript figure scripts (F1-F5)
  underdev/           In-progress figure scripts
si_scripts/           Supplementary information scripts (SI0-SI16)
  underdev/           In-progress SI scripts
methods/              Core library (generation, simulation, post-processing, analysis)
  metrics/            Shortfall and satisficing calculations
  plotting/           Publication figure utilities
data/                 Input data (climate change scenarios)
outputs/              Config-specific output root
  <config_name>/
    pywrdrb_inputs/   Ensemble input data (config-specific)
    data/
      simulations/    HDF5 simulation results
      models/         ModelBuilder JSON files
      drought_metrics/
      performance_metrics/
      event_metrics/
      zone_probabilities/
      satisficing/
      focal_events/
    figures/           Generated figures
    config.json        Configuration snapshot
docs/                 Manuscript and supplemental drafts
```
