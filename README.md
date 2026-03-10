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
| `04_postprocess_data_mpi.py` | Calculate shortage and contribution metrics |
| `05_calculate_ssi_drought_metrics.py` | Calculate SSI-based drought metrics (3, 6, 12-month windows) |
| `06_calculate_satisficing_by_drought.py` | Evaluate satisficing conditions during drought/non-drought periods |
| `07_calculate_storage_zone_probabilities.py` | Calculate reservoir storage zone probabilities and percentiles |

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
| `S3_postprocess_all.sh` | Submit postprocessing for all 3 datasets in parallel | Launches 3 SLURM jobs |
| `S3_postprocess_dataset.sh` | Postprocess a single dataset (called by `S3_postprocess_all.sh`) | 1 node, 20 tasks |
| `S4_calculate_ssi.sh` | Calculate SSI drought metrics, satisficing, and event metrics | 8 nodes, 40 tasks/node |
| `S5_run_figure_generation.sh` | Generate manuscript figures | 1 node |
| `S6_run_SI_scripts.sh` | Generate supplementary information figures | 1 node |

Usage:
```bash
# Full pipeline
sbatch S0_run_baseline_historic.sh
sbatch S1_run_stationary_ensemble.sh
sbatch S2_run_climate_adjusted_ensemble.sh
bash S3_postprocess_all.sh                              # submits 3 parallel jobs
sbatch S4_calculate_ssi.sh
sbatch S5_run_figure_generation.sh
sbatch S6_run_SI_scripts.sh

# Or postprocess a single dataset
sbatch S3_postprocess_dataset.sh stationary_ensemble
```

## File naming conventions

- `S*.sh` -- SLURM job submission scripts for HPC execution
- `0*.py` -- Main workflow scripts (generation, simulation, post-processing)
- `F*.py` -- Manuscript figure generation scripts
- `SI*.py` -- Supplementary information figure scripts

## Configuration

All ensemble and experiment parameters are defined in `methods/config.py`, including:

- Number of realizations and ensemble sets
- Simulation period and temporal resolution
- Dataset definitions (stationary and climate-adjusted scenarios)
- Pywr-DRB batching and output settings

## Project structure

```
methods/              Core library (generation, simulation, post-processing, analysis)
  metrics/            Shortfall and satisficing calculations
  plotting/           Publication figure utilities
data/                 Input data (climate change scenarios)
pywrdrb/              Pywr-DRB model files, inputs, and outputs
figures/              Generated figures
docs/                 Manuscript and supplemental drafts
```
