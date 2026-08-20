"""
Figure 3: Selected CMIP6 climate scenarios for the manuscript.

Local adaptation of CMIP6_multimodel_streamflow/S4_plot_scenarios.py with:
  - Both the CMIP6 target monthly % changes (dashed) and the actual achieved
    % changes realized by the climate-adjusted synthetic ensembles (solid),
    the latter taken from the cached SI18 validation table
    (outputs/{CONFIG_NAME}/data/climate_validation/nyc_monthly_pct_diff.csv).
  - Larger y-axis label fontsize.

All inputs are pre-computed CSVs; no flow-change calculations are re-run.

Usage:
    python plotting_scripts/Fig3_selected_climate_scenarios.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from methods.config import ROOT_DIR, CONFIG_DIR, FIG_DIR


SCENARIO_LABELS = {
    'low': 'Wetter Winter, Drier Summer',
    'medium': 'Median',
    'high': 'Wetter Winter',
    'historic': 'Historic Baseline'
}

ENSEMBLE_LABEL = 'Range of PRMS SSP2 RCP4.5 CMIP6 Models'

MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
SCENARIO_COLORS = {
    'low': '#ed9f1c',
    'medium': '#fee090',
    'high': '#009e73',
    'historic': '#000000'
}

TARGET_LINE_LABEL = 'CMIP6 Scenario (Target)'
ACHIEVED_LINE_LABEL = 'Achieved (Synthetic Ensemble)'

YLABEL_FONTSIZE = 14


def calculate_average_changes(monthly_pct_change_df):
    """Equal-weight average monthly % change for each scenario (column)."""
    return monthly_pct_change_df.mean(axis=0)


def plot_selected_scenarios(selected_scenarios_df,
                            achieved_changes_df,
                            all_scenarios_unfiltered_df,
                            node,
                            hydro_model,
                            ssp_period,
                            output_dir,
                            scenarios_to_show=None):
    """
    Create a two-panel figure showing selected climate scenarios:
    - Panel a: Rank vs magnitude plot of average annual flow changes
    - Panel b: Selected scenarios in context of full ensemble (monthly % changes),
      showing both CMIP6 target (dashed) and achieved ensemble (solid) changes.

    Parameters:
    -----------
    selected_scenarios_df : pd.DataFrame
        Selected scenarios (columns: low, medium, high) with target monthly % changes
    achieved_changes_df : pd.DataFrame
        Achieved monthly % changes realized by the climate-adjusted ensembles
        (columns matching selected scenario names, rows = months 1-12)
    all_scenarios_unfiltered_df : pd.DataFrame
        All scenarios before IQR filtering (pre-IQR) for full ensemble shading
    node : str
        Node name
    hydro_model : str
        Hydrologic model name (PRMS or VIC)
    ssp_period : str
        SSP period (e.g., '2020_2059')
    output_dir : str
        Output directory for figure
    scenarios_to_show : list or None
        List of scenario types to display (e.g., ['low', 'high']).
        If None, defaults to all scenarios in selected_scenarios_df.
    """
    if scenarios_to_show is None:
        scenarios_to_show = list(selected_scenarios_df.columns)

    average_changes_unfiltered = calculate_average_changes(all_scenarios_unfiltered_df)

    # =========================================================================
    # Consistent style settings for both panels
    # =========================================================================
    ENSEMBLE_COLOR = '#808080'
    ENSEMBLE_ALPHA = 0.4

    # Create figure with 2 panels, width ratio 1:1.67
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                    gridspec_kw={'width_ratios': [1, 1.67], 'wspace': 0.3})

    # ===== PANEL a: Rank vs Magnitude plot =====
    sorted_averages = average_changes_unfiltered.sort_values()
    ranks = np.arange(1, len(sorted_averages) + 1)

    ax1.plot(ranks, sorted_averages.values, color=ENSEMBLE_COLOR, linewidth=2,
             marker='o', markersize=6, alpha=0.7, label=ENSEMBLE_LABEL, zorder=1)

    # Highlight selected scenarios with colored markers
    for scenario_name, rank in zip(sorted_averages.index, ranks):
        for scenario_type in scenarios_to_show:
            if scenario_type in selected_scenarios_df.columns:
                scenario_monthly = selected_scenarios_df[scenario_type]
                if scenario_name in all_scenarios_unfiltered_df.columns:
                    if np.allclose(all_scenarios_unfiltered_df[scenario_name].values, scenario_monthly.values):
                        ax1.plot(rank, sorted_averages[scenario_name],
                                marker='o', markersize=10, color=SCENARIO_COLORS[scenario_type],
                                zorder=10, label=SCENARIO_LABELS[scenario_type])
                        break

    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

    ax1.set_xlabel('Scenario Rank', fontsize=11)
    ax1.set_ylabel('Change in Mean Annual Flow (%)', fontsize=YLABEL_FONTSIZE)
    ax1.set_title('(a) Ranked Annual Flow Changes',
                  fontsize=12, loc='left')
    ax1.grid(True, alpha=0.3, axis='y')

    # ===== PANEL b: Monthly % changes (ensemble context) =====
    unfiltered_min = all_scenarios_unfiltered_df.min(axis=1)
    unfiltered_max = all_scenarios_unfiltered_df.max(axis=1)
    ax2.fill_between(range(1, 13), unfiltered_min.values, unfiltered_max.values,
                     alpha=ENSEMBLE_ALPHA, color=ENSEMBLE_COLOR,
                     label=ENSEMBLE_LABEL)

    # Overlay selected scenarios: CMIP6 target (dashed) and achieved (solid)
    for scenario_type in scenarios_to_show:
        if scenario_type in selected_scenarios_df.columns:
            ax2.plot(range(1, 13), selected_scenarios_df[scenario_type].values,
                    linestyle='--', linewidth=2.5,
                    color=SCENARIO_COLORS[scenario_type],
                    label=SCENARIO_LABELS[scenario_type])
        if scenario_type in achieved_changes_df.columns:
            ax2.plot(range(1, 13), achieved_changes_df[scenario_type].values,
                    linestyle='-', marker='o', linewidth=2.5, markersize=6,
                    color=SCENARIO_COLORS[scenario_type])

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Month', fontsize=11)
    ax2.set_ylabel('Change in Mean Monthly Flow (%)', fontsize=YLABEL_FONTSIZE)
    ax2.set_title('(b) Monthly Flow Changes',
                  fontsize=12, loc='left')
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(MONTH_LABELS, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')

    # ===== Single unified legend below figure =====
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    legend_dict = {}
    for h, l in zip(handles1 + handles2, labels1 + labels2):
        if l not in legend_dict:
            legend_dict[l] = h

    # Order legend items: ensemble first, then scenarios, then line styles
    ordered_labels = []
    ordered_handles = []
    if ENSEMBLE_LABEL in legend_dict:
        ordered_labels.append(ENSEMBLE_LABEL)
        ordered_handles.append(legend_dict[ENSEMBLE_LABEL])
    for scenario_type in scenarios_to_show:
        label = SCENARIO_LABELS.get(scenario_type, scenario_type)
        if label in legend_dict:
            ordered_labels.append(label)
            ordered_handles.append(legend_dict[label])

    # Line-style entries distinguishing target vs achieved
    ordered_handles.append(Line2D([0], [0], color='black', linestyle='--', linewidth=2.5))
    ordered_labels.append(TARGET_LINE_LABEL)
    ordered_handles.append(Line2D([0], [0], color='black', linestyle='-', marker='o',
                                  linewidth=2.5, markersize=6))
    ordered_labels.append(ACHIEVED_LINE_LABEL)

    fig.legend(ordered_handles, ordered_labels,
              loc='upper center', bbox_to_anchor=(0.5, -0.02),
              ncol=3, fontsize=10, frameon=True)

    fname = f'{output_dir}/{node}_selected_scenarios_{hydro_model}_{ssp_period}.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {fname}")


def main():
    node = 'nyc_inflow'
    hydro_model_source = 'PRMS'
    ssp_period = '2020_2059'

    print(f"\n{'='*80}")
    print(f"FIG 3: SELECTED CLIMATE SCENARIOS")
    print(f"{'='*80}")
    print(f"Node:            {node}")
    print(f"Hydro Model:     {hydro_model_source}")
    print(f"Period:          {ssp_period}")
    print(f"{'='*80}\n")

    data_dir = f'{ROOT_DIR}/data'
    figures_dir = f'{FIG_DIR}/Fig3_selected_climate_scenarios'
    os.makedirs(figures_dir, exist_ok=True)

    # Selected scenario targets (same CSV used by methods.config for the
    # climate-adjusted ensemble generation)
    selected_scenarios_file = f'{data_dir}/{node}_selected_scenarios_{hydro_model_source}_{ssp_period}.csv'
    selected_scenarios_df = pd.read_csv(selected_scenarios_file, index_col=0)
    print(f"Loaded selected scenarios from: {selected_scenarios_file}")
    print(f"  Scenarios: {list(selected_scenarios_df.columns)}")

    # Achieved monthly % changes from the SI18 validation cache
    achieved_file = f'{CONFIG_DIR}/data/climate_validation/nyc_monthly_pct_diff.csv'
    if not os.path.exists(achieved_file):
        print(f"ERROR: Achieved-change table not found: {achieved_file}")
        print(f"Please run si_scripts/SI18_climate_adjustment_validation.py first.")
        return
    achieved_table = pd.read_csv(achieved_file, index_col='month')
    achieved_changes_df = pd.DataFrame({
        'low': achieved_table['low_pct_actual'],
        'high': achieved_table['high_pct_actual'],
    })
    print(f"Loaded achieved ensemble changes from: {achieved_file}")

    # Full CMIP6 ensemble monthly % change data (copied from the
    # CMIP6_multimodel_streamflow sibling repository)
    monthly_prc_change_file = f'{data_dir}/{node}_monthly_mean_prc_change_by_dataset_ssp_and_period.csv'
    monthly_prc_change = pd.read_csv(monthly_prc_change_file, index_col=0)

    # Filter for the same datasets used in scenario selection
    filtered_datasets = [d for d in monthly_prc_change.columns
                        if hydro_model_source in d
                        and ('ssp245' in d or 'ssp370' in d)
                        and ssp_period in d]
    all_scenarios_unfiltered_df = monthly_prc_change[filtered_datasets].copy()
    print(f"Full ensemble (pre-IQR): {len(all_scenarios_unfiltered_df.columns)} scenarios")

    plot_selected_scenarios(
        selected_scenarios_df,
        achieved_changes_df,
        all_scenarios_unfiltered_df,
        node,
        hydro_model_source,
        ssp_period,
        figures_dir,
        scenarios_to_show=['low', 'high']
    )

    print(f"\nFigure saved to: {figures_dir}")


if __name__ == "__main__":
    main()
