"""
Analyze shortage occurrence rates inside vs outside identified events.

This script validates whether the event identification is capturing
the periods where shortages occur.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '.')

from methods.vulnerability.config import VulnerabilityConfig
from methods.postprocess import (
    load_episode_analysis_data,
    preprocess_to_weekly,
    compute_weekly_climatology,
    add_standardized_variables,
)


def analyze_event_coverage(dataset_id: str = 'stationary_ensemble'):
    """
    Analyze how well events capture shortage occurrences.

    Compares:
    1. Shortage rate during event periods
    2. Shortage rate during non-event periods
    """
    print("=" * 70)
    print("EVENT COVERAGE ANALYSIS: Shortage Occurrence Inside vs Outside Events")
    print("=" * 70)
    print()

    # Load configuration
    config = VulnerabilityConfig(dataset_id=dataset_id)

    # Load events data (use get_output_path which includes dataset_id subdirectory)
    events_path = config.get_output_path('_events.parquet')
    events = pd.read_parquet(events_path)
    print(f"Loaded {len(events)} events from {events_path}")

    # Load weekly timeseries data using the standard approach
    data = load_episode_analysis_data(dataset_id)
    weekly_ts = preprocess_to_weekly(data, dataset_id, config)
    print(f"Loaded weekly timeseries with {len(weekly_ts)} rows")
    print(f"Columns: {weekly_ts.columns.tolist()[:10]}...")
    print()

    # Define shortage condition
    # Shortage = FFMP zone 5 (Emergency) or very low storage
    # Let's check what columns we have for this
    print("Weekly TS columns:", weekly_ts.columns.tolist())
    print()

    # Check if we have FFMP zone data
    if 'ffmp_zone' in weekly_ts.columns:
        # Shortage when FFMP zone >= 5 (Emergency/Drought Watch) or storage_pct < critical
        weekly_ts['is_shortage_week'] = weekly_ts['ffmp_zone'] >= 5
    elif 'storage_pct' in weekly_ts.columns:
        # Fallback: use storage_pct < 50% as critical threshold
        weekly_ts['is_shortage_week'] = weekly_ts['storage_pct'] < 50
    else:
        print("ERROR: Cannot identify shortage weeks - no ffmp_zone or storage_pct column")
        return

    # Mark which weeks are part of events
    weekly_ts['in_event'] = False

    # Check column names for realization and week index
    realization_col = 'realization' if 'realization' in weekly_ts.columns else 'realization_id'
    week_idx_col = 'week_idx' if 'week_idx' in weekly_ts.columns else 'week'

    print(f"Using columns: realization='{realization_col}', week_idx='{week_idx_col}'")
    print(f"Events columns: {events.columns.tolist()}")

    # Check column name for realization in events
    events_realization_col = 'realization' if 'realization' in events.columns else 'realization_id'

    for _, event in events.iterrows():
        realization = event[events_realization_col]
        start_week = event['start_week']
        end_week = event['end_week']

        # Mark weeks in this event
        mask = (weekly_ts[realization_col] == realization) & \
               (weekly_ts[week_idx_col] >= start_week) & \
               (weekly_ts[week_idx_col] <= end_week)
        weekly_ts.loc[mask, 'in_event'] = True

    # Calculate statistics
    in_event_weeks = weekly_ts[weekly_ts['in_event']]
    outside_event_weeks = weekly_ts[~weekly_ts['in_event']]

    n_in_event = len(in_event_weeks)
    n_outside_event = len(outside_event_weeks)

    shortages_in_event = in_event_weeks['is_shortage_week'].sum()
    shortages_outside_event = outside_event_weeks['is_shortage_week'].sum()

    shortage_rate_in_event = shortages_in_event / n_in_event if n_in_event > 0 else 0
    shortage_rate_outside_event = shortages_outside_event / n_outside_event if n_outside_event > 0 else 0

    # Calculate coverage metrics
    total_shortage_weeks = weekly_ts['is_shortage_week'].sum()
    coverage = shortages_in_event / total_shortage_weeks if total_shortage_weeks > 0 else 0

    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print("### Week Counts ###")
    print(f"Total weeks in dataset: {len(weekly_ts):,}")
    print(f"  - Weeks during events: {n_in_event:,} ({100*n_in_event/len(weekly_ts):.1f}%)")
    print(f"  - Weeks outside events: {n_outside_event:,} ({100*n_outside_event/len(weekly_ts):.1f}%)")
    print()

    print("### Shortage Occurrence ###")
    print(f"Total shortage weeks: {total_shortage_weeks:,}")
    print(f"  - During events: {shortages_in_event:,}")
    print(f"  - Outside events: {shortages_outside_event:,}")
    print()

    print("### Shortage Rates ###")
    print(f"Shortage rate DURING events: {100*shortage_rate_in_event:.2f}%")
    print(f"Shortage rate OUTSIDE events: {100*shortage_rate_outside_event:.2f}%")
    print(f"Ratio (during/outside): {shortage_rate_in_event/shortage_rate_outside_event:.1f}x"
          if shortage_rate_outside_event > 0 else "N/A (no shortages outside events)")
    print()

    print("### Event Capture Rate ###")
    print(f"% of all shortage weeks captured by events: {100*coverage:.1f}%")
    print(f"% of shortage weeks MISSED (outside events): {100*(1-coverage):.1f}%")
    print()

    # Additional breakdown by shortage severity (if possible)
    if 'ffmp_zone' in weekly_ts.columns:
        print("### Breakdown by FFMP Zone ###")
        for zone in sorted(weekly_ts['ffmp_zone'].unique()):
            zone_weeks = weekly_ts[weekly_ts['ffmp_zone'] == zone]
            in_events = zone_weeks['in_event'].sum()
            total = len(zone_weeks)
            print(f"FFMP Zone {zone}: {in_events:,} / {total:,} weeks in events ({100*in_events/total:.1f}%)")

    print()

    # Quality assessment
    print("=" * 70)
    print("ASSESSMENT")
    print("=" * 70)
    if coverage >= 0.90:
        print("[EXCELLENT] Events capture >=90% of shortage weeks")
    elif coverage >= 0.75:
        print("[GOOD] Events capture >=75% of shortage weeks")
    elif coverage >= 0.50:
        print("[MODERATE] Events capture >=50% of shortage weeks - may need refinement")
    else:
        print("[POOR] Events capture <50% of shortage weeks - event definition needs revision")

    if shortage_rate_in_event > 5 * shortage_rate_outside_event:
        print("[EXCELLENT] Shortage rate during events is >5x higher than outside")
    elif shortage_rate_in_event > 2 * shortage_rate_outside_event:
        print("[GOOD] Shortage rate during events is 2-5x higher than outside")
    else:
        print("[MODERATE] Events may be too broad or poorly targeted")

    print()

    # Save results
    results = {
        'total_weeks': len(weekly_ts),
        'weeks_in_events': n_in_event,
        'weeks_outside_events': n_outside_event,
        'total_shortage_weeks': int(total_shortage_weeks),
        'shortages_in_events': int(shortages_in_event),
        'shortages_outside_events': int(shortages_outside_event),
        'shortage_rate_in_events': shortage_rate_in_event,
        'shortage_rate_outside_events': shortage_rate_outside_event,
        'shortage_capture_rate': coverage,
    }

    results_path = config.get_output_path('_event_coverage_analysis.txt')
    with open(results_path, 'w') as f:
        f.write("EVENT COVERAGE ANALYSIS RESULTS\n")
        f.write("=" * 50 + "\n\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")

    print(f"Results saved to: {results_path}")

    return results


if __name__ == "__main__":
    analyze_event_coverage()
