# Stochastic Exploratory Experiment

Code for the stochastic drought experiment in the Delaware River Basin (DRB) described in:

> Amestoy, T. and Reed, P. (2026). Exploratory Modeling to Better Understand How Changing Drought
> Extremes May Shape Water Supply Vulnerabilities in the Delaware River Basin. Submitted to
> *Journal of Hydrology: Regional Studies*.

The experiment generates three ensembles of 2,000 synthetic daily streamflow realizations (70 water
years each, 2030 to 2100), simulates DRB reservoir operations for every realization with
[Pywr-DRB](https://github.com/Pywr-DRB/Pywr-DRB), identifies drought events with the Standardized
Streamflow Index (SSI), and evaluates New York City (NYC) water supply outcomes under the Flexible
Flow Management Program (FFMP) rules.

| `dataset_id` | Manuscript name | Description |
|---|---|---|
| `stationary_ensemble` | Stationary Baseline | Resamples the 1945 to 2023 historical variability with no climate adjustment |
| `climate_adjusted_low` | WWDS | Wetter winters, drier summers: monthly mean-flow changes from the selected CMIP6-driven PRMS projection |
| `climate_adjusted_high` | WWSS | Wetter winters, similar summers |

Synthetic flows are generated with the Kirsch monthly generator and Nowak daily disaggregation
implemented in [SynHydro](https://github.com/TrevorJA/SynHydro) at 22 Pywr-DRB nodes, with eight
minor gauge nodes filled by regression on their upstream node. The Kirsch generator is fit to the
1945 to 2023 Pywr-DRB historical reconstruction (`pub_nhmv10_BC_withObsScaled`), with its monthly
means and standard deviations taken from the 1980 to 2019 baseline period; the Nowak disaggregation
is fit to the full record. Climate adjustment shifts the baseline monthly means by the percentage
changes in `data/nyc_inflow_selected_scenarios_PRMS_2020_2059.csv`.

## Requirements

Python 3.11 with the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

- [Pywr-DRB](https://github.com/Pywr-DRB/Pywr-DRB) v2.2.0 ([Zenodo](https://doi.org/10.5281/zenodo.21716508)).
  Supplies the historical streamflow reconstruction, the diversion records, and the `perfect_foresight`
  flow-prediction mode used by this experiment.
- [SynHydro](https://github.com/TrevorJA/SynHydro) v0.1.0 ([Zenodo](https://doi.org/10.5281/zenodo.22759484)).
  Kirsch-Nowak generation and SSI drought identification.
- [mpi4py](https://mpi4py.readthedocs.io/) and an MPI installation for the parallel steps.

The experiment was run on the Hopper cluster at Cornell with SLURM; the `S*.sh` scripts hold the job
settings that were used (Python 3.11.5, OpenMPI 4.0.5).

## Data

- `data/nyc_inflow_selected_scenarios_PRMS_2020_2059.csv`: monthly percentage changes in mean
  aggregate NYC inflow for the selected climate scenarios (`low` = WWDS, `high` = WWSS; `medium` was
  not used).
- `data/nyc_inflow_monthly_mean_prc_change_by_dataset_ssp_and_period.csv`: the same statistic for all
  CMIP6-driven projections considered during scenario selection (SI Table S1.1); kept for reference,
  not read by the code. Both files were produced with
  [CMIP6_multimodel_streamflow](https://github.com/Pywr-DRB/CMIP6_multimodel_streamflow).
- Historical streamflow and diversion records are read from the Pywr-DRB package.
- The synthetic ensembles and the derived metric tables produced by this code are archived on Zenodo
  (DOI to be added).

## Workflow

Each numbered script is one step. Steps 01 to 06 take a `dataset_id` argument and run under MPI
(`mpirun -np N python <script> <dataset_id>`) or serially; step 05 also accepts `historic` to
identify the drought events of the historical record.

| Script | Step |
|---|---|
| `00_run_baseline_simulations.py` | Simulate the historical reconstruction with Pywr-DRB |
| `01_generate_ensemble_sets.py` | Generate synthetic daily flows (Kirsch-Nowak), 20 sets of 100 realizations per ensemble |
| `02_prep_pywrdrb_inputs.py` | Build the Pywr-DRB inputs for each set (predicted inflows, NYC and NJ diversion demands) |
| `03_run_pywrdrb_simulations.py` | Simulate every realization with Pywr-DRB |
| `04_postprocess_data_mpi.py` | Compute shortages and NYC release contributions from the simulation output |
| `05_calculate_ssi_drought_metrics.py` | Identify SSI-3, SSI-6 and SSI-12 drought events |
| `06_calculate_performance_metrics.py` | Annual metrics, Hashimoto reliability and resiliency, storage-zone events, per-event metrics |
| `extract_manuscript_values.py` | Collect the numerical values quoted in the manuscript into `manuscript/ensemble_stats/` |

SLURM submission scripts:

| Script | Runs |
|---|---|
| `S0_run_baseline_historic.sh` | Step 00 |
| `S1_run_stationary_ensemble.sh` | Steps 01 to 03 for the Stationary Baseline |
| `S2_run_climate_adjusted_ensemble.sh` | Steps 01 to 03 for WWDS and WWSS |
| `S3_postprocess_all.sh` | Step 04 for all ensembles |
| `S4_calculate_ssi.sh` | Step 05 for the historical record and all ensembles |
| `S5_calculate_performance_metrics.sh` | Step 06 for all ensembles and the reconstruction, then FFMP storage-zone probabilities (`si_scripts/SI3_...`) |
| `S6_run_figure_generation.sh` | Manuscript figures |
| `S7_run_SI_scripts.sh` | Supplementary figures |
| `S8_extract_manuscript_values.sh` | `extract_manuscript_values.py` |
| `S99_run_entire_workflow.sh` | Submits S0 to S8 as one SLURM dependency chain |

Job order in `S99_run_entire_workflow.sh`:

```
S0 baseline
 |
S1 generate --> S1 prep -----> S1 simulate
      \---> S2 generate ----/       |
                              S2 prep --> S2 simulate
                                              |
                                        S3 postprocess
                                              |
                                        S4 SSI droughts
                                              |
                                        S5 performance metrics
                                              |
                              S6 figures | S7 SI figures | S8 manuscript values
```

The generation and simulation steps run on several nodes; the Pywr-DRB input preparation runs on one
node because the Pywr-DRB v2.2 ensemble preprocessors stall over multi-node TCP at this scale.

## Configuration

All experiment settings are in `methods/config.py`: ensemble size and set structure, simulation period,
dataset definitions, SSI windows, NYC reservoir storage capacities, and the severity-magnitude grid and
thresholds that define the consequential drought region. `CONFIG_NAME` selects the output directory
`outputs/<CONFIG_NAME>/`; the published results use `perf_foresight_fullkirsch`. A snapshot of the
settings is written to `outputs/<CONFIG_NAME>/config.json` by step 01.

```
outputs/<CONFIG_NAME>/
  config.json
  pywrdrb_inputs/<dataset_id>/<dataset_id>_set<N>/   synthetic flows and Pywr-DRB inputs per set (N = 1 to 20)
  data/
    simulations/          reconstruction.hdf5, Pywr-DRB output per set, one post-processed HDF5 per
                          ensemble, and cache/ (focal-region setup shared by Fig10 and SI22)
    models/               Pywr-DRB model JSON files
    drought_metrics/      SSI drought events and the SI20 duration-severity table
    performance_metrics/  annual, Hashimoto, contribution and storage-zone metrics
    event_metrics/        metrics per SSI drought event
    zone_probabilities/   FFMP storage-zone probabilities by week
    climate_validation/   realized monthly flow changes of the climate-adjusted ensembles
    satisficing/, focal_events/   created by step 01, unused
  figures/
  SI22_scenario_storyline_droughts/   SI22 figures and selected events
```

## Figures

| Script | Figure |
|---|---|
| `plotting_scripts/Fig4_plot_ensemble_diagnostics.py` | SI Figure S2.5 |
| `plotting_scripts/Fig5_plot_drought_metric_distribution.py` | Figure 4 (SSI-3); SI Figures S3.2 and S3.3 (SSI-6, SSI-12) |
| `plotting_scripts/Fig6_plot_drought_zone_occurrence.py` | Figure 5 |
| `plotting_scripts/Fig7_2x3_kdes_3mo_9mo_windows.py` | SI Figure S5.1 |
| `plotting_scripts/Fig8_plot_contribution_distributions.py` | Figure 6 |
| `plotting_scripts/Fig9_plot_drought_satisficing_heatmap_2col.py` | Figure 7 |
| `plotting_scripts/Fig10_focal_drought_dynamics.py` | Figure 8 |
| `si_scripts/SI0_full_ensemble_diagnostics.py` | SI Figures S2.6 to S2.8 |
| `si_scripts/SI0b_low_flow_convergence.py` | SI Figure S2.9 |
| `si_scripts/SI2_plot_diversion_diagnostics.py` | SI Figures S2.2 to S2.4 |
| `si_scripts/SI15_plot_ssi_window_emergency_capture.py` | SI Figure S3.1 |
| `si_scripts/SI18_climate_adjustment_validation.py` | SI Figure S2.1 |
| `si_scripts/SI19_drought_return_period_convergence.py` | SI Figures S2.10 to S2.12 |
| `si_scripts/SI20_duration_severity_curves.py` | SI Figure S3.4 |
| `si_scripts/SI22_scenario_storyline_droughts.py` | SI Figures S6.1 to S6.3 |

Figures 1 (basin map), 2 (methods schematic) and 3 (climate scenario selection, from
[CMIP6_multimodel_streamflow](https://github.com/Pywr-DRB/CMIP6_multimodel_streamflow)) are not
produced by this repository. The script numbers predate the final figure numbering of the manuscript.

## Repository layout

```
0*_*.py                 Workflow steps
S*.sh                   SLURM submission scripts
extract_manuscript_values.py
methods/                Shared code
  config.py             Experiment settings
  ensemble_utils.py     Ensemble set specifications, directories and the config.json snapshot
  mpi_utils.py          Rank assignment and point-to-point MPI helpers
  generate.py           Kirsch-Nowak ensemble generation
  prepare.py            Pywr-DRB input preparation
  simulate.py           Pywr-DRB batch simulation
  load.py               Readers for inputs, simulation output and metric files
  postprocess.py        Performance metric calculations
  drought_analysis.py   SSI fitting and historical drought events
  return_period.py      Joint-exceedance drought return periods
  storage_zones.py      FFMP storage-zone probabilities
  zone_duration_metrics.py  Drought-zone episodes
  contribution.py       NYC release contribution series
  water_year.py, utils.py, verification.py, print_summary.py, save.py   small helpers
  metrics/              Shortage, contribution and per-event metric calculations
  plotting/             Shared figure styles and plotting functions
plotting_scripts/       Manuscript figure scripts
si_scripts/             Supplementary figure scripts
data/                   Climate scenario inputs
```
