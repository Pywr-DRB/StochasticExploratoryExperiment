"""
F12: Drought Comparison Timeseries Figure.

Compares two water years from the stationary ensemble that have similar
drought hazard (SSI severity/magnitude) but different satisficing outcomes.

Layout: 3x2 grid
  - Rows: Storage, Flow+Contribution, Satisficing Metrics
  - Columns: Pass drought (left), Fail drought (right)
  - Each column shows actual dates for its drought period

Usage:
    python F12_plot_drought_comparison_timeseries.py

Generates 5 figures with different selection schemes.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

import pywrdrb
from methods.config import FIG_DIR, NYC_RESERVOIRS, NYC_TOTAL_CAPACITY
from methods.plotting.styles import (
    DPI_HIGH, FONTSIZE_SMALL, FONTSIZE_MEDIUM, FONTSIZE_LABEL,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

FIG_OUTPUT_DIR = f"{FIG_DIR}/F12_drought_comparison"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

SATISFICING_DATA_DIR = "./pywrdrb/satisficing_analysis"
DROUGHT_METRICS_DIR = "./pywrdrb/drought_metrics"
SSI_WINDOW = 12
DATASET_ID = 'stationary_ensemble'

# Buffer days before/after drought period
BUFFER_DAYS = 90

# Colors
COLOR_PASS = '#009E73'   # Bluish green
COLOR_FAIL = '#D55E00'   # Vermilion/orange
COLOR_SECONDARY = '#0072B2'  # Blue for secondary metrics
COLOR_THRESHOLD = 'black'

# Thresholds
STORAGE_THRESHOLD = 20.0  # %
VIOLATION_DAYS_THRESHOLD = 3  # days

# Pair selection thresholds
SEVERITY_TOLERANCE = 0.5   # Absolute difference (relaxed)
MAGNITUDE_TOLERANCE = 0.40  # Relative difference (40%, relaxed)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_drought_data_with_satisficing():
    """
    Load drought events with correct simulation dates and merge with satisficing.

    The drought_events.csv has actual simulation dates (2030-2099).
    The years_with_droughts.csv has satisficing outcomes by year.
    We merge them to get drought events with satisficing info.
    """
    # Load drought events (has correct simulation dates)
    events_fname = f"{DROUGHT_METRICS_DIR}/{DATASET_ID}_ssi{SSI_WINDOW}_drought_events.csv"
    events_df = pd.read_csv(events_fname)
    events_df['start'] = pd.to_datetime(events_df['start'])
    events_df['end'] = pd.to_datetime(events_df['end'])
    events_df['start_year'] = events_df['start'].dt.year

    # Load satisficing data (by year)
    satisficing_fname = f"{SATISFICING_DATA_DIR}/{DATASET_ID}_ssi{SSI_WINDOW}_years_with_droughts.csv"
    satisficing_df = pd.read_csv(satisficing_fname)

    # Rename columns for merge
    if 'realization' in satisficing_df.columns:
        satisficing_df = satisficing_df.rename(columns={'realization': 'realization_id'})

    # Merge on year and realization
    merged = events_df.merge(
        satisficing_df[['year', 'realization_id', 'min_storage_pct', 'max_violation_days', 'satisficing']],
        left_on=['start_year', 'realization_id'],
        right_on=['year', 'realization_id'],
        how='left'
    )

    # Drop rows without satisficing data
    merged = merged.dropna(subset=['min_storage_pct'])

    # Add satisficing category
    merged['storage_pass'] = merged['min_storage_pct'] >= STORAGE_THRESHOLD
    merged['montague_pass'] = merged['max_violation_days'] <= VIOLATION_DAYS_THRESHOLD

    print(f"  Loaded {len(merged)} drought events with satisficing data")
    print(f"  Date range: {merged['start'].min()} to {merged['end'].max()}")
    print(f"  Satisficing pass: {merged['satisficing'].sum()}, fail: {(~merged['satisficing']).sum()}")

    return merged


def find_contrasting_pair(df, scheme='closest_severity'):
    """
    Find two droughts with similar hazard but different outcomes.

    Parameters
    ----------
    df : pd.DataFrame
        Drought events with satisficing data
    scheme : str
        Selection scheme:
        - 'closest_severity': Best severity match
        - 'extreme_contrast': Maximum storage difference
        - 'moderate_drought': Mid-severity droughts only
        - 'long_duration': Prefer longer duration droughts
        - 'different_realizations': Force different realizations
    """
    df_pass = df[df['satisficing'] == True].copy()
    df_fail = df[df['satisficing'] == False].copy()

    print(f"  Pass droughts: {len(df_pass)}, Fail droughts: {len(df_fail)}")
    print(f"  Selection scheme: {scheme}")

    if len(df_fail) == 0:
        print("  No failed droughts found, selecting by storage difference")
        df_sorted = df.sort_values('min_storage_pct')
        return df_sorted.iloc[-1], df_sorted.iloc[0]

    # Different selection schemes
    if scheme == 'closest_severity':
        # Original: find closest severity match
        best_pair = None
        best_score = float('inf')
        for _, fail_row in df_fail.iterrows():
            for _, pass_row in df_pass.iterrows():
                sev_diff = abs(pass_row['severity'] - fail_row['severity'])
                if sev_diff > SEVERITY_TOLERANCE:
                    continue
                mag_ref = max(abs(pass_row['magnitude']), abs(fail_row['magnitude']), 1)
                mag_diff = abs(pass_row['magnitude'] - fail_row['magnitude']) / mag_ref
                if mag_diff > MAGNITUDE_TOLERANCE:
                    continue
                storage_diff = pass_row['min_storage_pct'] - fail_row['min_storage_pct']
                score = sev_diff - storage_diff / 20
                if score < best_score:
                    best_score = score
                    best_pair = (pass_row, fail_row)
        if best_pair:
            return best_pair

    elif scheme == 'extreme_contrast':
        # Maximum storage contrast with reasonable severity match
        best_pair = None
        best_contrast = 0
        for _, fail_row in df_fail.iterrows():
            for _, pass_row in df_pass.iterrows():
                sev_diff = abs(pass_row['severity'] - fail_row['severity'])
                if sev_diff > SEVERITY_TOLERANCE * 1.5:  # Slightly relaxed
                    continue
                storage_diff = pass_row['min_storage_pct'] - fail_row['min_storage_pct']
                if storage_diff > best_contrast:
                    best_contrast = storage_diff
                    best_pair = (pass_row, fail_row)
        if best_pair:
            return best_pair

    elif scheme == 'moderate_drought':
        # Select mid-severity droughts (not extreme)
        median_sev = df['severity'].median()
        df_pass_mod = df_pass[df_pass['severity'].between(median_sev - 1, median_sev + 0.5)]
        df_fail_mod = df_fail[df_fail['severity'].between(median_sev - 1, median_sev + 0.5)]
        if len(df_pass_mod) > 0 and len(df_fail_mod) > 0:
            pass_row = df_pass_mod.loc[df_pass_mod['min_storage_pct'].idxmax()]
            fail_row = df_fail_mod.loc[df_fail_mod['min_storage_pct'].idxmin()]
            return pass_row, fail_row

    elif scheme == 'long_duration':
        # Prefer longer duration droughts
        min_duration = df['duration'].quantile(0.5)  # Above median
        df_pass_long = df_pass[df_pass['duration'] >= min_duration]
        df_fail_long = df_fail[df_fail['duration'] >= min_duration]
        if len(df_pass_long) > 0 and len(df_fail_long) > 0:
            best_pair = None
            best_score = float('inf')
            for _, fail_row in df_fail_long.iterrows():
                for _, pass_row in df_pass_long.iterrows():
                    sev_diff = abs(pass_row['severity'] - fail_row['severity'])
                    if sev_diff > SEVERITY_TOLERANCE * 1.5:
                        continue
                    score = sev_diff
                    if score < best_score:
                        best_score = score
                        best_pair = (pass_row, fail_row)
            if best_pair:
                return best_pair

    elif scheme == 'different_realizations':
        # Force selection from different realizations
        best_pair = None
        best_score = float('inf')
        for _, fail_row in df_fail.iterrows():
            for _, pass_row in df_pass.iterrows():
                if pass_row['realization_id'] == fail_row['realization_id']:
                    continue  # Force different realizations
                sev_diff = abs(pass_row['severity'] - fail_row['severity'])
                if sev_diff > SEVERITY_TOLERANCE * 1.2:
                    continue
                storage_diff = pass_row['min_storage_pct'] - fail_row['min_storage_pct']
                score = sev_diff - storage_diff / 30
                if score < best_score:
                    best_score = score
                    best_pair = (pass_row, fail_row)
        if best_pair:
            return best_pair

    # Fallback
    print(f"  No matching pair found for scheme '{scheme}', using fallback")
    best_pass = df_pass.loc[df_pass['min_storage_pct'].idxmax()]
    worst_fail = df_fail.loc[df_fail['min_storage_pct'].idxmin()]
    return best_pass, worst_fail


# Available selection schemes
SELECTION_SCHEMES = [
    'closest_severity',
    'extreme_contrast',
    'moderate_drought',
    'long_duration',
    'different_realizations',
]


def load_drought_timeseries(realization_id, start_date, end_date):
    """
    Load timeseries data for a drought period from HDF5.
    """
    fname = f'./pywrdrb/outputs/{DATASET_ID}_with_postprocessing.hdf5'

    data = pywrdrb.Data()
    data.load_from_export(fname, results_sets=[
        'res_storage', 'major_flow', 'contribution', 'shortage',
        'ibt_diversions', 'ibt_demands'
    ])

    r = realization_id

    # Calculate plot range with buffer
    plot_start = pd.Timestamp(start_date) - pd.Timedelta(days=BUFFER_DAYS)
    plot_end = pd.Timestamp(end_date) + pd.Timedelta(days=BUFFER_DAYS)

    # NYC aggregate storage (%)
    storage_df = data.res_storage[DATASET_ID][r]
    nyc_storage = storage_df[NYC_RESERVOIRS].sum(axis=1)
    nyc_storage_pct = 100 * nyc_storage / NYC_TOTAL_CAPACITY

    # Montague flow
    montague_flow = data.major_flow[DATASET_ID][r]['delMontague']

    # NYC contribution to Montague
    nyc_contribution = data.contribution[DATASET_ID][r]['mrf_montagueTrenton_nyc']

    # Contribution as % of Montague flow (avoid div by zero)
    contribution_pct = 100 * nyc_contribution / montague_flow.clip(lower=1)

    # Montague shortage (violation indicator)
    montague_shortage = data.shortage[DATASET_ID][r]['delMontague']

    # NYC diversions and demands
    nyc_diversion = data.ibt_diversions[DATASET_ID][r]['delivery_nyc']
    nyc_demand = data.ibt_demands[DATASET_ID][r]['demand_nyc']
    nyc_shortage = (nyc_demand - nyc_diversion).clip(lower=0)

    # Filter to plot range
    def filter_range(series):
        filtered = series.loc[plot_start:plot_end]
        return filtered

    result = {
        'storage_pct': filter_range(nyc_storage_pct),
        'montague_flow': filter_range(montague_flow),
        'contribution': filter_range(nyc_contribution),
        'contribution_pct': filter_range(contribution_pct),
        'montague_shortage': filter_range(montague_shortage),
        'nyc_diversion': filter_range(nyc_diversion),
        'nyc_demand': filter_range(nyc_demand),
        'nyc_shortage': filter_range(nyc_shortage),
        'drought_start': pd.Timestamp(start_date),
        'drought_end': pd.Timestamp(end_date),
    }

    print(f"    Loaded {len(result['storage_pct'])} days of data")

    return result


def calculate_consecutive_violations(shortage_series):
    """
    Calculate rolling count of consecutive days with shortage > 0.
    """
    is_violation = (shortage_series > 0).astype(int)
    groups = (~is_violation.astype(bool)).cumsum()
    consecutive = is_violation.groupby(groups).cumsum()
    return consecutive


# ============================================================================
# PLOTTING
# ============================================================================

def plot_storage_row(axes, data_pass, data_fail, drought_pass, drought_fail):
    """Plot Row 1: NYC Storage panels."""
    for ax, data, drought, color, title_suffix in [
        (axes[0], data_pass, drought_pass, COLOR_PASS, 'Pass'),
        (axes[1], data_fail, drought_fail, COLOR_FAIL, 'Fail'),
    ]:
        # Storage line
        ax.plot(data['storage_pct'].index, data['storage_pct'].values,
                color=color, linewidth=1.5)

        # Threshold line
        ax.axhline(STORAGE_THRESHOLD, color=COLOR_THRESHOLD,
                   linestyle='--', linewidth=1, alpha=0.7)

        # Formatting
        start_str = str(drought['start'])[:10]
        end_str = str(drought['end'])[:10]
        sev = drought['severity']
        min_stor = drought['min_storage_pct']
        ax.set_title(f"{start_str[:7]} to {end_str[:7]} ({title_suffix})\n"
                     f"Severity: {sev:.2f}, Min Storage: {min_stor:.1f}%",
                     fontsize=FONTSIZE_MEDIUM)

        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

    axes[0].set_ylabel('NYC Storage (%)', fontsize=FONTSIZE_LABEL)


def plot_flow_row(axes, data_pass, data_fail):
    """Plot Row 2: Montague Flow + Contribution % (dual y-axis)."""
    for ax, data, color in [
        (axes[0], data_pass, COLOR_PASS),
        (axes[1], data_fail, COLOR_FAIL),
    ]:
        # Left axis: Montague flow
        ax.plot(data['montague_flow'].index, data['montague_flow'].values,
                color=color, linewidth=1.2, label='Montague Flow')

        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Right axis: Contribution %
        ax2 = ax.twinx()
        ax2.plot(data['contribution_pct'].index, data['contribution_pct'].values,
                 color=COLOR_SECONDARY, linewidth=1, linestyle='--',
                 alpha=0.8, label='NYC Contrib %')
        ax2.set_ylim(0, 100)

        if ax == axes[1]:
            ax2.set_ylabel('NYC Contribution (%)', fontsize=FONTSIZE_SMALL,
                          color=COLOR_SECONDARY)
            ax2.tick_params(axis='y', labelcolor=COLOR_SECONDARY)
        else:
            ax2.set_yticklabels([])

    axes[0].set_ylabel('Montague Flow (MGD)', fontsize=FONTSIZE_LABEL)


def plot_satisficing_row(axes, data_pass, data_fail):
    """Plot Row 3: Consecutive violations + NYC shortage (dual y-axis)."""
    for ax, data, color in [
        (axes[0], data_pass, COLOR_PASS),
        (axes[1], data_fail, COLOR_FAIL),
    ]:
        # Calculate consecutive violations
        violations = calculate_consecutive_violations(data['montague_shortage'])

        # Left axis: Consecutive violation days
        ax.plot(violations.index, violations.values,
                color=color, linewidth=1.2, label='Consec. Violations')

        # Threshold line
        ax.axhline(VIOLATION_DAYS_THRESHOLD, color=COLOR_THRESHOLD,
                   linestyle='--', linewidth=1, alpha=0.7)

        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Right axis: NYC shortage
        ax2 = ax.twinx()
        ax2.fill_between(data['nyc_shortage'].index, 0, data['nyc_shortage'].values,
                         color=COLOR_SECONDARY, alpha=0.3, label='NYC Shortage')
        ax2.plot(data['nyc_shortage'].index, data['nyc_shortage'].values,
                 color=COLOR_SECONDARY, linewidth=0.8, alpha=0.6)

        if ax == axes[1]:
            ax2.set_ylabel('NYC Shortage (MGD)', fontsize=FONTSIZE_SMALL,
                          color=COLOR_SECONDARY)
            ax2.tick_params(axis='y', labelcolor=COLOR_SECONDARY)
        else:
            ax2.set_yticklabels([])

    axes[0].set_ylabel('Consecutive\nViolation Days', fontsize=FONTSIZE_LABEL)


def create_shared_legend(fig):
    """Create shared legend at bottom of figure."""
    handles = [
        Line2D([0], [0], color=COLOR_PASS, linewidth=2, label='Pass Drought'),
        Line2D([0], [0], color=COLOR_FAIL, linewidth=2, label='Fail Drought'),
        Line2D([0], [0], color=COLOR_SECONDARY, linewidth=1.5, linestyle='--',
               label='NYC Contrib % / Shortage'),
        Line2D([0], [0], color=COLOR_THRESHOLD, linewidth=1, linestyle='--',
               label='Satisficing Threshold'),
    ]

    fig.legend(handles=handles, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, 0.01), fontsize=FONTSIZE_SMALL,
               frameon=True, fancybox=True)


# ============================================================================
# MAIN
# ============================================================================

def generate_figure(df, scheme, scheme_idx):
    """Generate figure for a specific selection scheme."""
    print(f"\n{'=' * 60}")
    print(f"Scheme {scheme_idx + 1}/{len(SELECTION_SCHEMES)}: {scheme}")
    print(f"{'=' * 60}")

    print("\nFinding contrasting drought pair...")
    drought_pass, drought_fail = find_contrasting_pair(df, scheme=scheme)

    print(f"\nSelected droughts:")
    print(f"  PASS: realization={int(drought_pass['realization_id'])}, "
          f"{str(drought_pass['start'])[:10]} to {str(drought_pass['end'])[:10]}")
    print(f"        severity={drought_pass['severity']:.2f}, "
          f"magnitude={drought_pass['magnitude']:.1f}, "
          f"min_storage={drought_pass['min_storage_pct']:.1f}%")
    print(f"  FAIL: realization={int(drought_fail['realization_id'])}, "
          f"{str(drought_fail['start'])[:10]} to {str(drought_fail['end'])[:10]}")
    print(f"        severity={drought_fail['severity']:.2f}, "
          f"magnitude={drought_fail['magnitude']:.1f}, "
          f"min_storage={drought_fail['min_storage_pct']:.1f}%, "
          f"violations={int(drought_fail['max_violation_days'])} days")

    # Load timeseries data
    print("\nLoading timeseries data...")
    print("  Loading pass drought data...")
    data_pass = load_drought_timeseries(
        int(drought_pass['realization_id']),
        drought_pass['start'], drought_pass['end']
    )
    print("  Loading fail drought data...")
    data_fail = load_drought_timeseries(
        int(drought_fail['realization_id']),
        drought_fail['start'], drought_fail['end']
    )

    # Create figure
    print("\nCreating figure...")
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # Plot each row
    plot_storage_row(axes[0], data_pass, data_fail, drought_pass, drought_fail)
    plot_flow_row(axes[1], data_pass, data_fail)
    plot_satisficing_row(axes[2], data_pass, data_fail)

    # Format x-axes (only bottom row shows labels)
    for row in range(2):
        for col in range(2):
            axes[row, col].tick_params(labelbottom=False)

    for col in range(2):
        axes[2, col].set_xlabel('Date', fontsize=FONTSIZE_LABEL)
        axes[2, col].tick_params(axis='x', rotation=45)

    # Add row labels on left
    row_labels = ['NYC Storage', 'Montague Flow', 'Satisficing']
    for row, label in enumerate(row_labels):
        axes[row, 0].annotate(
            label, xy=(-0.18, 0.5), xycoords='axes fraction',
            fontsize=FONTSIZE_MEDIUM,
            ha='center', va='center', rotation=90,
        )

    # Shared legend
    create_shared_legend(fig)

    # Overall title with scheme name
    scheme_labels = {
        'closest_severity': 'Closest Severity Match',
        'extreme_contrast': 'Maximum Storage Contrast',
        'moderate_drought': 'Moderate Severity Droughts',
        'long_duration': 'Longer Duration Droughts',
        'different_realizations': 'Different Realizations',
    }
    scheme_label = scheme_labels.get(scheme, scheme)

    fig.suptitle(
        f'Comparing Droughts with Similar Severity but Different Outcomes\n'
        f'({scheme_label})',
        fontsize=14, fontweight='bold', y=0.99
    )

    plt.tight_layout(rect=[0.05, 0.06, 1, 0.94])

    # Save with scheme in filename
    fname = f"{FIG_OUTPUT_DIR}/F12_drought_comparison_ssi{SSI_WINDOW}_{scheme}.png"
    plt.savefig(fname, dpi=DPI_HIGH, bbox_inches='tight')
    print(f"\nSaved: {fname}")

    plt.close()

    return fname


def main():
    """Main entry point."""
    print("=" * 70)
    print("F12: Drought Comparison Timeseries")
    print(f"Running {len(SELECTION_SCHEMES)} selection schemes")
    print("=" * 70)

    # Load drought data with satisficing (once)
    print("\nLoading drought data with satisficing...")
    df = load_drought_data_with_satisficing()

    # Generate figure for each selection scheme
    saved_files = []
    for idx, scheme in enumerate(SELECTION_SCHEMES):
        fname = generate_figure(df, scheme, idx)
        saved_files.append(fname)

    # Summary
    print("\n" + "=" * 70)
    print("ALL FIGURES GENERATED!")
    print("=" * 70)
    print("\nSaved files:")
    for fname in saved_files:
        print(f"  - {fname}")
    print("\nDone!")


if __name__ == "__main__":
    main()
