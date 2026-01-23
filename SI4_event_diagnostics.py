"""
Event Diagnostics: Coverage Visualization and Contrasting Event Analysis.

This script:
1. Visualizes event coverage analysis results
2. Identifies pairs of similar events with contrasting outcomes
3. Creates detailed diagnostic plots comparing these events
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '.')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns

from methods.vulnerability.config import VulnerabilityConfig
from methods.postprocess import (
    load_episode_analysis_data,
    preprocess_to_weekly,
)

# Style settings
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11


def plot_event_coverage_summary(dataset_id: str = 'stationary_ensemble'):
    """
    Create visualizations for event coverage analysis.

    Shows:
    1. Shortage rate comparison (in vs out of events)
    2. FFMP zone capture rates
    3. Shortage distribution pie chart
    """
    config = VulnerabilityConfig(dataset_id=dataset_id)
    fig_dir = config.output_dir / config.dataset_id / 'figures'
    fig_dir.mkdir(exist_ok=True)

    # Load events and weekly data
    events_path = config.get_output_path('_events.parquet')
    events = pd.read_parquet(events_path)

    data = load_episode_analysis_data(dataset_id)
    weekly_ts = preprocess_to_weekly(data, dataset_id, config)

    # Mark event weeks
    weekly_ts['in_event'] = False
    realization_col = 'realization_id'
    week_col = 'week'

    for _, event in events.iterrows():
        mask = (weekly_ts[realization_col] == event['realization_id']) & \
               (weekly_ts[week_col] >= event['start_week']) & \
               (weekly_ts[week_col] <= event['end_week'])
        weekly_ts.loc[mask, 'in_event'] = True

    # Define shortage
    weekly_ts['is_shortage'] = weekly_ts['ffmp_zone'] >= 5

    # Calculate statistics
    in_event = weekly_ts[weekly_ts['in_event']]
    outside_event = weekly_ts[~weekly_ts['in_event']]

    shortage_rate_in = in_event['is_shortage'].mean() * 100
    shortage_rate_out = outside_event['is_shortage'].mean() * 100

    total_shortages = weekly_ts['is_shortage'].sum()
    shortages_in = in_event['is_shortage'].sum()
    shortages_out = outside_event['is_shortage'].sum()

    # FFMP zone capture rates
    ffmp_capture = {}
    for zone in sorted(weekly_ts['ffmp_zone'].unique()):
        zone_data = weekly_ts[weekly_ts['ffmp_zone'] == zone]
        capture_rate = zone_data['in_event'].mean() * 100
        ffmp_capture[int(zone)] = capture_rate

    # Create figure
    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1.2, 1])

    # Panel 1: Shortage rate comparison
    ax1 = fig.add_subplot(gs[0])

    bars = ax1.bar(['During\nEvents', 'Outside\nEvents'],
                   [shortage_rate_in, shortage_rate_out],
                   color=['#e74c3c', '#3498db'], edgecolor='black', linewidth=1)

    ax1.set_ylabel('Shortage Rate (%)')
    ax1.set_title('Shortage Occurrence Rate', fontweight='bold')

    # Add value labels
    for bar, val in zip(bars, [shortage_rate_in, shortage_rate_out]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add ratio annotation
    ratio = shortage_rate_in / shortage_rate_out if shortage_rate_out > 0 else np.inf
    ax1.text(0.5, 0.95, f'Ratio: {ratio:.1f}x', transform=ax1.transAxes,
            ha='center', va='top', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel 2: FFMP zone capture rates
    ax2 = fig.add_subplot(gs[1])

    zones = list(ffmp_capture.keys())
    rates = list(ffmp_capture.values())

    # Color gradient based on capture rate
    colors = plt.cm.RdYlGn([r/100 for r in rates])

    bars = ax2.bar([str(z) for z in zones], rates, color=colors, edgecolor='black', linewidth=0.5)

    ax2.set_xlabel('FFMP Zone')
    ax2.set_ylabel('Event Capture Rate (%)')
    ax2.set_title('Event Capture by FFMP Zone', fontweight='bold')
    ax2.axhline(50, color='gray', linestyle='--', alpha=0.7, label='50% threshold')

    # Add zone descriptions
    zone_labels = {0: 'Surplus', 1: 'Normal', 2: 'Caution',
                   3: 'Warning', 4: 'Watch', 5: 'Emergency', 6: 'Drought'}

    for bar, zone, rate in zip(bars, zones, rates):
        label = zone_labels.get(zone, '')
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.0f}%', ha='center', va='bottom', fontsize=9)

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_ylim(0, 105)

    # Panel 3: Shortage distribution pie
    ax3 = fig.add_subplot(gs[2])

    sizes = [shortages_in, shortages_out]
    labels = [f'Captured\n({shortages_in})', f'Missed\n({shortages_out})']
    colors_pie = ['#27ae60', '#e74c3c']
    explode = (0.02, 0.05)

    wedges, texts, autotexts = ax3.pie(
        sizes, labels=labels, colors=colors_pie, explode=explode,
        autopct='%1.1f%%', startangle=90,
        wedgeprops=dict(edgecolor='black', linewidth=1)
    )
    autotexts[0].set_fontweight('bold')
    autotexts[1].set_fontweight('bold')

    ax3.set_title('Shortage Week Capture', fontweight='bold')

    plt.suptitle('Event Coverage Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = fig_dir / f'{dataset_id}_event_coverage_summary.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")

    plt.close(fig)

    return weekly_ts


def find_contrasting_event_pair(dataset_id: str = 'stationary_ensemble'):
    """
    Find a pair of similar events with contrasting outcomes.

    Algorithm:
    1. Filter events in high-shortage-probability zone (top consensus features)
    2. Separate into multi-failure and zero-failure groups
    3. For each multi-failure event, find most similar zero-failure event
       based on Euclidean distance in feature space
    4. Return the pair with smallest distance (most similar but different outcomes)

    Returns
    -------
    tuple
        (multi_failure_event, zero_failure_event, similarity_score)
    """
    config = VulnerabilityConfig(dataset_id=dataset_id)

    # Load event features
    features_path = config.get_output_path('_event_features.parquet')
    event_features = pd.read_parquet(features_path)

    print(f"\nTotal events: {len(event_features)}")

    # Define high-risk zone based on decision rules:
    # Rule 1: forcing_imbalance_during > 1.90 AND week_of_year_onset > 43.50 -> 84% shortage
    # Rule 3: forcing_imbalance_during > 1.90 AND week_of_year_onset <= 43.50 -> 59% shortage
    # So high-risk = forcing_imbalance_during > 1.5 (slightly relaxed threshold)

    high_risk_mask = event_features['forcing_imbalance_during'] > 1.5
    high_risk_events = event_features[high_risk_mask].copy()
    print(f"High-risk zone events (forcing_imbalance > 1.5): {len(high_risk_events)}")

    # Calculate total failure count
    failure_cols = ['storage_failure', 'demand_failure', 'flow_failure']
    high_risk_events['n_failures'] = high_risk_events[failure_cols].sum(axis=1)

    # Separate groups
    multi_failure = high_risk_events[high_risk_events['n_failures'] >= 2].copy()
    zero_failure = high_risk_events[high_risk_events['n_failures'] == 0].copy()

    print(f"Multi-failure events (>=2 failures): {len(multi_failure)}")
    print(f"Zero-failure events (0 failures): {len(zero_failure)}")

    if len(multi_failure) == 0 or len(zero_failure) == 0:
        print("Cannot find contrasting pair - relaxing criteria...")
        # Relax to any failure vs no failure
        multi_failure = high_risk_events[high_risk_events['n_failures'] >= 1].copy()
        print(f"Any-failure events (>=1 failures): {len(multi_failure)}")

    if len(multi_failure) == 0 or len(zero_failure) == 0:
        print("Still no valid pairs - using full dataset")
        high_risk_events = event_features.copy()
        high_risk_events['n_failures'] = high_risk_events[failure_cols].sum(axis=1)
        multi_failure = high_risk_events[high_risk_events['n_failures'] >= 1].copy()
        zero_failure = high_risk_events[high_risk_events['n_failures'] == 0].copy()

    # Features for similarity calculation (exogenous features only)
    similarity_features = [
        'forcing_imbalance_during',
        'inflow_std_cum_12wk_pre',
        'week_of_year_onset',
        'storage_pct_onset',
        'inflow_std_mean_during',
    ]

    # Normalize features for distance calculation
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    all_features = pd.concat([multi_failure[similarity_features],
                              zero_failure[similarity_features]])
    scaler.fit(all_features)

    multi_scaled = scaler.transform(multi_failure[similarity_features])
    zero_scaled = scaler.transform(zero_failure[similarity_features])

    # Find most similar pair
    best_distance = np.inf
    best_multi_idx = None
    best_zero_idx = None

    for i, (mi, mf) in enumerate(zip(multi_failure.index, multi_scaled)):
        for j, (zi, zf) in enumerate(zip(zero_failure.index, zero_scaled)):
            dist = np.sqrt(np.sum((mf - zf) ** 2))
            if dist < best_distance:
                best_distance = dist
                best_multi_idx = mi
                best_zero_idx = zi

    multi_event = event_features.loc[best_multi_idx]
    zero_event = event_features.loc[best_zero_idx]

    print(f"\n=== Best Contrasting Pair Found ===")
    print(f"Similarity distance: {best_distance:.3f}")
    print(f"\nMulti-failure event (ID={best_multi_idx}):")
    print(f"  Realization: {multi_event['realization_id']}")
    print(f"  Week: {multi_event['start_week']} - {multi_event['end_week']}")
    print(f"  Failures: storage={multi_event['storage_failure']}, "
          f"demand={multi_event['demand_failure']}, flow={multi_event['flow_failure']}")
    print(f"  forcing_imbalance: {multi_event['forcing_imbalance_during']:.2f}")
    print(f"  outcome: {multi_event['outcome']}")

    print(f"\nZero-failure event (ID={best_zero_idx}):")
    print(f"  Realization: {zero_event['realization_id']}")
    print(f"  Week: {zero_event['start_week']} - {zero_event['end_week']}")
    print(f"  Failures: storage={zero_event['storage_failure']}, "
          f"demand={zero_event['demand_failure']}, flow={zero_event['flow_failure']}")
    print(f"  forcing_imbalance: {zero_event['forcing_imbalance_during']:.2f}")
    print(f"  outcome: {zero_event['outcome']}")

    # Feature comparison
    print(f"\n=== Feature Comparison ===")
    print(f"{'Feature':<30} {'Multi-Fail':>12} {'Zero-Fail':>12} {'Diff':>10}")
    print("-" * 66)
    for feat in similarity_features:
        mval = multi_event[feat]
        zval = zero_event[feat]
        diff = mval - zval
        print(f"{feat:<30} {mval:>12.2f} {zval:>12.2f} {diff:>10.2f}")

    return multi_event, zero_event, best_distance


def plot_event_pair_diagnostics(
    multi_event: pd.Series,
    zero_event: pd.Series,
    dataset_id: str = 'stationary_ensemble',
):
    """
    Create detailed multi-panel diagnostic plots comparing two events.

    Panels:
    1. Combined NYC reservoir storage (%)
    2. NYC demand and diversions
    3. NYC contributions to Montague and Montague flow vs target
    """
    config = VulnerabilityConfig(dataset_id=dataset_id)
    fig_dir = config.output_dir / config.dataset_id / 'figures'

    # Load full data
    data = load_episode_analysis_data(dataset_id)

    # We need to load the raw daily/weekly data for these specific realizations
    # Load from the postprocessed HDF5 file
    from pywrdrb import Data
    fname = f'./pywrdrb/outputs/{dataset_id}_with_postprocessing.hdf5'

    full_data = Data()
    full_data.load_from_export(fname, results_sets=[
        'res_storage', 'res_release', 'major_flow', 'mrf_target',
        'nyc_release_components', 'demand'
    ])

    # Get realization IDs
    multi_real = multi_event['realization_id']
    zero_real = zero_event['realization_id']

    events_data = []
    for event, label, color in [
        (multi_event, 'Multi-Failure', '#e74c3c'),
        (zero_event, 'Zero-Failure', '#3498db')
    ]:
        real_id = event['realization_id']
        start_week = int(event['start_week'])
        end_week = int(event['end_week'])

        # Convert week to approximate date range
        # Assuming weeks start from simulation start
        # We'll extract a window around the event
        buffer_weeks = 8  # weeks before and after

        # Get storage data
        storage_df = full_data.res_storage[dataset_id][real_id]

        # NYC reservoirs
        nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']
        nyc_storage = storage_df[nyc_reservoirs].sum(axis=1)

        # Convert to weekly and get relevant window
        # First, create week index
        storage_df['week'] = (storage_df.index - storage_df.index[0]).days // 7

        event_mask = (storage_df['week'] >= start_week - buffer_weeks) & \
                     (storage_df['week'] <= end_week + buffer_weeks)

        event_data = {
            'label': label,
            'color': color,
            'event': event,
            'storage': nyc_storage[event_mask],
            'dates': storage_df.index[event_mask],
            'start_week': start_week,
            'end_week': end_week,
        }

        # Get demand data
        if dataset_id in full_data.demand and real_id in full_data.demand[dataset_id]:
            demand_df = full_data.demand[dataset_id][real_id]
            if 'week' not in demand_df.columns:
                demand_df = demand_df.copy()
                demand_df['week'] = (demand_df.index - demand_df.index[0]).days // 7

            event_mask_demand = (demand_df['week'] >= start_week - buffer_weeks) & \
                               (demand_df['week'] <= end_week + buffer_weeks)

            # NYC demand - look for nyc columns
            nyc_demand_cols = [c for c in demand_df.columns if 'nyc' in c.lower()]
            if nyc_demand_cols:
                event_data['nyc_demand'] = demand_df[nyc_demand_cols].sum(axis=1)[event_mask_demand]

        # Get release data (diversions)
        if dataset_id in full_data.res_release and real_id in full_data.res_release[dataset_id]:
            release_df = full_data.res_release[dataset_id][real_id]
            if 'week' not in release_df.columns:
                release_df = release_df.copy()
                release_df['week'] = (release_df.index - release_df.index[0]).days // 7

            event_mask_rel = (release_df['week'] >= start_week - buffer_weeks) & \
                            (release_df['week'] <= end_week + buffer_weeks)

            if nyc_reservoirs[0] in release_df.columns:
                nyc_release = release_df[nyc_reservoirs].sum(axis=1)
                event_data['nyc_release'] = nyc_release[event_mask_rel]
                event_data['release_dates'] = release_df.index[event_mask_rel]

        # Get Montague flow and target
        if dataset_id in full_data.major_flow and real_id in full_data.major_flow[dataset_id]:
            flow_df = full_data.major_flow[dataset_id][real_id]
            if 'week' not in flow_df.columns:
                flow_df = flow_df.copy()
                flow_df['week'] = (flow_df.index - flow_df.index[0]).days // 7

            event_mask_flow = (flow_df['week'] >= start_week - buffer_weeks) & \
                             (flow_df['week'] <= end_week + buffer_weeks)

            if 'delMontague' in flow_df.columns:
                event_data['montague_flow'] = flow_df['delMontague'][event_mask_flow]
                event_data['flow_dates'] = flow_df.index[event_mask_flow]

        if dataset_id in full_data.mrf_target and real_id in full_data.mrf_target[dataset_id]:
            target_df = full_data.mrf_target[dataset_id][real_id]
            if 'week' not in target_df.columns:
                target_df = target_df.copy()
                target_df['week'] = (target_df.index - target_df.index[0]).days // 7

            event_mask_tgt = (target_df['week'] >= start_week - buffer_weeks) & \
                            (target_df['week'] <= end_week + buffer_weeks)

            if 'delMontague' in target_df.columns:
                event_data['montague_target'] = target_df['delMontague'][event_mask_tgt]

        # Get NYC contributions to Montague
        if hasattr(full_data, 'nyc_release_components') and \
           dataset_id in full_data.nyc_release_components and \
           real_id in full_data.nyc_release_components[dataset_id]:
            comp_df = full_data.nyc_release_components[dataset_id][real_id]
            if 'week' not in comp_df.columns:
                comp_df = comp_df.copy()
                comp_df['week'] = (comp_df.index - comp_df.index[0]).days // 7

            event_mask_comp = (comp_df['week'] >= start_week - buffer_weeks) & \
                             (comp_df['week'] <= end_week + buffer_weeks)

            # Look for Montague contribution columns
            mrf_cols = [c for c in comp_df.columns if 'mrf' in c.lower() or 'montague' in c.lower()]
            if mrf_cols:
                event_data['nyc_montague_contrib'] = comp_df[mrf_cols].sum(axis=1)[event_mask_comp]

        events_data.append(event_data)

    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex='col')

    for col, edata in enumerate(events_data):
        event = edata['event']
        label = edata['label']
        color = edata['color']

        # Normalize x-axis to weeks relative to event start
        start_week = edata['start_week']
        end_week = edata['end_week']

        # Panel 1: Storage
        ax = axes[0, col]
        if 'storage' in edata and len(edata['storage']) > 0:
            dates = edata['dates']
            storage = edata['storage']

            # Convert to percent of capacity
            nyc_capacity = 272.6 + 140.2 + 34.9  # BG from config
            storage_pct = (storage / nyc_capacity) * 100

            # Calculate relative weeks
            days_from_start = (dates - dates[0]).days
            weeks = days_from_start / 7

            ax.plot(weeks, storage_pct, color=color, linewidth=1.5)
            ax.fill_between(weeks, storage_pct, alpha=0.3, color=color)

            # Mark event window
            event_start_rel = start_week - (start_week - 8)
            event_end_rel = end_week - (start_week - 8)
            ax.axvspan(event_start_rel, event_end_rel, alpha=0.2, color='gray', label='Event Window')

            # Add threshold lines
            ax.axhline(60, color='orange', linestyle='--', alpha=0.7, label='FFMP Warning')
            ax.axhline(40, color='red', linestyle='--', alpha=0.7, label='FFMP Emergency')

        ax.set_ylabel('NYC Storage (%)')
        ax.set_title(f'{label} Event\nReal={event["realization_id"]}, Weeks {start_week}-{end_week}',
                    fontweight='bold', color=color)
        ax.legend(loc='lower left', fontsize=8)
        ax.set_ylim(0, 100)

        # Panel 2: Demand and Diversions
        ax = axes[1, col]

        if 'nyc_release' in edata and len(edata['nyc_release']) > 0:
            dates = edata['release_dates']
            days_from_start = (dates - dates[0]).days
            weeks = days_from_start / 7

            ax.plot(weeks, edata['nyc_release'], color='blue', linewidth=1.5,
                   label='NYC Releases')

        if 'nyc_demand' in edata and len(edata['nyc_demand']) > 0:
            # Align demand dates with release dates
            ax.plot(weeks[:len(edata['nyc_demand'])],
                   edata['nyc_demand'].values[:len(weeks)],
                   color='purple', linewidth=1.5, linestyle='--',
                   label='NYC Demand')

        ax.set_ylabel('Flow (MGD)')
        ax.legend(loc='upper right', fontsize=8)

        # Panel 3: Montague flow and target
        ax = axes[2, col]

        if 'montague_flow' in edata and len(edata['montague_flow']) > 0:
            dates = edata['flow_dates']
            days_from_start = (dates - dates[0]).days
            weeks = days_from_start / 7

            ax.plot(weeks, edata['montague_flow'], color='green', linewidth=1.5,
                   label='Montague Flow')

            if 'montague_target' in edata and len(edata['montague_target']) > 0:
                ax.plot(weeks[:len(edata['montague_target'])],
                       edata['montague_target'].values[:len(weeks)],
                       color='red', linewidth=2, linestyle='--',
                       label='MRF Target')

                # Shade deficit
                flow = edata['montague_flow'].values
                target = edata['montague_target'].values[:len(flow)]
                if len(target) == len(flow):
                    ax.fill_between(weeks, flow, target,
                                   where=(flow < target),
                                   color='red', alpha=0.3, label='Shortage')

        ax.set_ylabel('Flow (MGD)')
        ax.set_xlabel('Weeks from Event Start - 8')
        ax.legend(loc='upper right', fontsize=8)

    # Add failure summary annotations
    for col, edata in enumerate(events_data):
        event = edata['event']
        failures = []
        if event['storage_failure']:
            failures.append('Storage')
        if event['demand_failure']:
            failures.append('Demand')
        if event['flow_failure']:
            failures.append('Flow')

        fail_text = ', '.join(failures) if failures else 'None'

        fig.text(0.25 + col * 0.5, 0.02,
                f'Failures: {fail_text} | Outcome: {event["outcome"]}',
                ha='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle('Contrasting Event Pair Diagnostics', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])

    output_path = fig_dir / f'{dataset_id}_contrasting_events_diagnostic.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_path}")

    plt.close(fig)

    return fig


def plot_simplified_event_comparison(
    multi_event: pd.Series,
    zero_event: pd.Series,
    dataset_id: str = 'stationary_ensemble',
):
    """
    Create simplified comparison using weekly preprocessed data.
    """
    config = VulnerabilityConfig(dataset_id=dataset_id)
    fig_dir = config.output_dir / config.dataset_id / 'figures'

    # Load weekly data
    data = load_episode_analysis_data(dataset_id)
    weekly_ts = preprocess_to_weekly(data, dataset_id, config)

    # Create figure
    fig, axes = plt.subplots(4, 2, figsize=(14, 14))

    events = [
        (multi_event, 'Multi-Failure', '#e74c3c'),
        (zero_event, 'Zero-Failure', '#3498db')
    ]

    for col, (event, label, color) in enumerate(events):
        real_id = event['realization_id']
        start_week = int(event['start_week'])
        end_week = int(event['end_week'])
        buffer = 12

        # Get data for this realization and time window
        mask = (weekly_ts['realization_id'] == real_id) & \
               (weekly_ts['week'] >= start_week - buffer) & \
               (weekly_ts['week'] <= end_week + buffer)

        event_data = weekly_ts[mask].copy()

        if len(event_data) == 0:
            print(f"No data found for {label} event")
            continue

        # Relative week
        event_data['rel_week'] = event_data['week'] - start_week
        x = event_data['rel_week']

        # Panel 1: Storage
        ax = axes[0, col]
        ax.plot(x, event_data['storage_pct'], color=color, linewidth=2)
        ax.fill_between(x, event_data['storage_pct'], alpha=0.3, color=color)
        ax.axvspan(0, end_week - start_week, alpha=0.15, color='gray')
        ax.axhline(60, color='orange', linestyle='--', alpha=0.7, label='Warning (60%)')
        ax.axhline(40, color='red', linestyle='--', alpha=0.7, label='Emergency (40%)')
        ax.set_ylabel('Storage (%)')
        ax.set_title(f'{label} Event\nReal={real_id}, Weeks {start_week}-{end_week}',
                    fontweight='bold', color=color)
        ax.legend(loc='lower left', fontsize=8)
        ax.set_ylim(0, 100)

        # Panel 2: FFMP Zone
        ax = axes[1, col]
        ax.step(x, event_data['ffmp_zone'], where='mid', color=color, linewidth=2)
        ax.axvspan(0, end_week - start_week, alpha=0.15, color='gray')
        ax.axhline(5, color='red', linestyle='--', alpha=0.7, label='Emergency threshold')
        ax.set_ylabel('FFMP Zone')
        ax.set_ylim(-0.5, 6.5)
        ax.set_yticks(range(7))
        ax.legend(loc='upper right', fontsize=8)

        # Panel 3: Demand satisfaction
        ax = axes[2, col]
        if 'demand_satisfaction' in event_data.columns:
            ax.plot(x, event_data['demand_satisfaction'] * 100, color=color, linewidth=2)
            ax.axvspan(0, end_week - start_week, alpha=0.15, color='gray')
            ax.axhline(100, color='green', linestyle='--', alpha=0.7)
            ax.axhline(95, color='orange', linestyle='--', alpha=0.7, label='95% threshold')
        ax.set_ylabel('Demand Satisfaction (%)')
        ax.set_ylim(80, 101)
        ax.legend(loc='lower left', fontsize=8)

        # Panel 4: Flow satisfaction
        ax = axes[3, col]
        if 'flow_satisfaction' in event_data.columns:
            ax.plot(x, event_data['flow_satisfaction'] * 100, color=color, linewidth=2)
            ax.axvspan(0, end_week - start_week, alpha=0.15, color='gray')
            ax.axhline(100, color='green', linestyle='--', alpha=0.7)
            ax.axhline(95, color='orange', linestyle='--', alpha=0.7, label='95% threshold')
        ax.set_ylabel('Flow Satisfaction (%)')
        ax.set_xlabel('Weeks Relative to Event Start')
        ax.set_ylim(80, 101)
        ax.legend(loc='lower left', fontsize=8)

    # Add summary text
    for col, (event, label, color) in enumerate(events):
        failures = []
        if event['storage_failure']:
            failures.append('Storage')
        if event['demand_failure']:
            failures.append(f'Demand ({int(event["max_demand_consec_days"])}d)')
        if event['flow_failure']:
            failures.append(f'Flow ({int(event["max_flow_consec_days"])}d)')

        fail_text = ', '.join(failures) if failures else 'None'

        fig.text(0.25 + col * 0.5, 0.01,
                f'Failures: {fail_text}\n'
                f'Forcing Imbalance: {event["forcing_imbalance_during"]:.2f} | '
                f'Antecedent Deficit: {event["inflow_std_cum_12wk_pre"]:.2f}',
                ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle('Contrasting Event Pair: System Response Comparison',
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.06, 1, 0.97])

    output_path = fig_dir / f'{dataset_id}_contrasting_events_simplified.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")

    plt.close(fig)


def plot_event_pair_in_feature_space(
    multi_event: pd.Series,
    zero_event: pd.Series,
    dataset_id: str = 'stationary_ensemble',
):
    """
    Show where the contrasting events fall in the decision tree feature space.
    """
    config = VulnerabilityConfig(dataset_id=dataset_id)
    fig_dir = config.output_dir / config.dataset_id / 'figures'

    # Load all event features
    features_path = config.get_output_path('_event_features.parquet')
    event_features = pd.read_parquet(features_path)

    # Key features from decision rules
    x_feat = 'forcing_imbalance_during'
    y_feat = 'inflow_std_cum_12wk_pre'

    fig, ax = plt.subplots(figsize=(10, 8))

    # Background: all events
    shortage_mask = event_features['outcome'] == 'shortage'

    ax.scatter(event_features.loc[~shortage_mask, x_feat],
              event_features.loc[~shortage_mask, y_feat],
              c='#3498db', marker='o', s=40, alpha=0.4,
              edgecolors='white', linewidths=0.3,
              label='Recovered')

    ax.scatter(event_features.loc[shortage_mask, x_feat],
              event_features.loc[shortage_mask, y_feat],
              c='#e74c3c', marker='s', s=50, alpha=0.4,
              edgecolors='white', linewidths=0.3,
              label='Shortage')

    # Decision boundaries from rules
    # Rule: forcing_imbalance > 1.90 -> high risk
    ax.axvline(1.90, color='black', linestyle='--', linewidth=2,
              label='Decision boundary')

    # Rule: inflow_std_cum_12wk_pre <= -2.41 -> shortage risk
    ax.axhline(-2.41, color='black', linestyle='--', linewidth=2)

    # Highlight contrasting pair
    ax.scatter(multi_event[x_feat], multi_event[y_feat],
              c='red', marker='*', s=400, edgecolors='black', linewidths=2,
              zorder=10, label='Multi-Failure Event')

    ax.scatter(zero_event[x_feat], zero_event[y_feat],
              c='lime', marker='*', s=400, edgecolors='black', linewidths=2,
              zorder=10, label='Zero-Failure Event')

    # Connect with line
    ax.plot([multi_event[x_feat], zero_event[x_feat]],
           [multi_event[y_feat], zero_event[y_feat]],
           'k-', linewidth=2, alpha=0.5)

    # Add annotations
    ax.annotate(f'Multi-Fail\n({multi_event["outcome"]})',
               xy=(multi_event[x_feat], multi_event[y_feat]),
               xytext=(10, 10), textcoords='offset points',
               fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))

    ax.annotate(f'Zero-Fail\n({zero_event["outcome"]})',
               xy=(zero_event[x_feat], zero_event[y_feat]),
               xytext=(10, -20), textcoords='offset points',
               fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lime', alpha=0.7))

    ax.set_xlabel('Forcing Imbalance During Event\n(demand_max - inflow_mean)', fontsize=12)
    ax.set_ylabel('Cumulative Inflow Anomaly (12wk pre-event)', fontsize=12)
    ax.set_title('Contrasting Events in Decision Space', fontsize=14, fontweight='bold')

    ax.legend(loc='upper left', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add region labels
    ax.text(3.5, 2, 'HIGH RISK\nREGION', ha='center', va='center',
           fontsize=12, fontweight='bold', color='red', alpha=0.5)
    ax.text(-1, 2, 'MODERATE\nRISK', ha='center', va='center',
           fontsize=11, color='orange', alpha=0.5)

    plt.tight_layout()

    output_path = fig_dir / f'{dataset_id}_contrasting_events_feature_space.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")

    plt.close(fig)


if __name__ == "__main__":
    dataset_id = 'stationary_ensemble'

    print("=" * 70)
    print("EVENT DIAGNOSTICS AND COVERAGE VISUALIZATION")
    print("=" * 70)

    # 1. Coverage visualization
    print("\n1. Creating event coverage visualizations...")
    weekly_ts = plot_event_coverage_summary(dataset_id)

    # 2. Find contrasting pair
    print("\n2. Finding contrasting event pair...")
    multi_event, zero_event, distance = find_contrasting_event_pair(dataset_id)

    # 3. Feature space visualization
    print("\n3. Plotting events in feature space...")
    plot_event_pair_in_feature_space(multi_event, zero_event, dataset_id)

    # 4. Simplified diagnostic comparison
    print("\n4. Creating simplified diagnostic comparison...")
    plot_simplified_event_comparison(multi_event, zero_event, dataset_id)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
