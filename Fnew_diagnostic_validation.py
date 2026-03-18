"""
Diagnostic validation: Confirm key patterns in the data that will drive figure design.

Tests:
1. Seasonal storage divergence across scenarios
2. Severity-outcome relationships (event-level)
3. Contribution amplification by severity bin
4. Zone probability timing shifts
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from methods.config import ROOT_DIR

DATASETS = ['stationary_ensemble', 'climate_adjusted_low', 'climate_adjusted_high']
LABELS = {
    'stationary_ensemble': 'Baseline',
    'climate_adjusted_low': 'Mixed Future',
    'climate_adjusted_high': 'Wet Future',
}

def load_event_metrics(dataset_id, ssi_window=6):
    fname = f'{ROOT_DIR}/pywrdrb/event_metrics/{dataset_id}_ssi{ssi_window}_event_metrics.csv'
    return pd.read_csv(fname)

def load_annual_metrics(dataset_id):
    fname = f'{ROOT_DIR}/pywrdrb/performance_metrics/{dataset_id}_annual_metrics.csv'
    return pd.read_csv(fname)

def load_storage_percentiles(dataset_id):
    fname = f'{ROOT_DIR}/pywrdrb/zone_probabilities/{dataset_id}_storage_percentiles_weekly.csv'
    return pd.read_csv(fname, index_col='period')

def load_zone_probs(dataset_id):
    fname = f'{ROOT_DIR}/pywrdrb/zone_probabilities/{dataset_id}_zone_probs_weekly.csv'
    return pd.read_csv(fname, index_col='period')

def load_contribution_metrics(dataset_id):
    fname = f'{ROOT_DIR}/pywrdrb/performance_metrics/{dataset_id}_contribution_metrics.csv'
    return pd.read_csv(fname)

print("=" * 80)
print("DIAGNOSTIC VALIDATION FOR FIGURE DESIGN")
print("=" * 80)

# ============================================================================
# TEST 1: Seasonal storage divergence
# ============================================================================
print("\n" + "=" * 80)
print("TEST 1: Seasonal Storage Divergence")
print("=" * 80)

storage_data = {}
for did in DATASETS:
    storage_data[did] = load_storage_percentiles(did)

# Compare median (p50) and 5th percentile across scenarios by week
print("\nWeekly storage comparison (Median / 5th percentile):")
print(f"{'Week':>6} | {'Stat p50':>10} {'Low p50':>10} {'High p50':>10} | {'Stat p5':>10} {'Low p5':>10} {'High p5':>10}")
print("-" * 80)

# Water year mapping: weeks 23-52 = June-Dec, weeks 1-22 = Jan-May
# Show key weeks: June (wk 23), Aug (wk 32), Oct (wk 40), Dec (wk 49), Feb (wk 6), Apr (wk 14)
key_weeks = [23, 27, 32, 36, 40, 44, 49, 1, 6, 10, 14, 18]
month_names = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May']

for wk, mo in zip(key_weeks, month_names):
    if wk in storage_data['stationary_ensemble'].index:
        vals_p50 = [storage_data[d].loc[wk, 'p50'] for d in DATASETS]
        vals_p5 = [storage_data[d].loc[wk, 'p5'] for d in DATASETS]
        print(f"{mo:>4} {wk:>2} | {vals_p50[0]:>10.1f} {vals_p50[1]:>10.1f} {vals_p50[2]:>10.1f} | {vals_p5[0]:>10.1f} {vals_p5[1]:>10.1f} {vals_p5[2]:>10.1f}")

# Maximum divergence
stat_p5 = storage_data['stationary_ensemble']['p5']
low_p5 = storage_data['climate_adjusted_low']['p5']
high_p5 = storage_data['climate_adjusted_high']['p5']

max_low_diff_week = (low_p5 - stat_p5).abs().idxmax()
max_high_diff_week = (high_p5 - stat_p5).abs().idxmax()
print(f"\nMax 5th-percentile divergence (low vs stat): Week {max_low_diff_week}, Delta = {(low_p5 - stat_p5).loc[max_low_diff_week]:.1f}%")
print(f"Max 5th-percentile divergence (high vs stat): Week {max_high_diff_week}, Delta = {(high_p5 - stat_p5).loc[max_high_diff_week]:.1f}%")

# ============================================================================
# TEST 2: Event-level severity-outcome relationships
# ============================================================================
print("\n" + "=" * 80)
print("TEST 2: Event-Level Severity-Outcome Relationships")
print("=" * 80)

events = {}
for did in DATASETS:
    events[did] = load_event_metrics(did, ssi_window=6)
    print(f"\n{LABELS[did]}: {len(events[did])} events")
    print(f"  Severity: mean={events[did]['severity'].mean():.2f}, p50={events[did]['severity'].median():.2f}, p95={events[did]['severity'].quantile(0.95):.2f}")
    print(f"  Duration (days): mean={events[did]['duration_days'].mean():.0f}, p50={events[did]['duration_days'].median():.0f}")
    print(f"  NYC contrib (MG): mean={events[did]['total_nyc_contribution_mg'].mean():.0f}, p50={events[did]['total_nyc_contribution_mg'].median():.0f}")
    print(f"  Min storage (%): mean={events[did]['event_min_storage_pct'].mean():.1f}, p50={events[did]['event_min_storage_pct'].median():.1f}")
    print(f"  Max consec Montague days: mean={events[did]['max_consec_montague_days'].mean():.1f}")
    print(f"  Diversion satisfaction: mean={events[did]['nyc_diversion_sat_ratio'].mean():.3f}")

# Bin events by severity and compare outcomes
severity_bins = [0, 1.0, 2.0, 10.0]
severity_labels = ['Mild (0-1)', 'Moderate (1-2)', 'Severe (2+)']

print("\n\nOutcomes by severity bin:")
for metric, metric_label in [
    ('event_min_storage_pct', 'Min Storage (%)'),
    ('total_nyc_contribution_mg', 'NYC Contribution (MG)'),
    ('max_consec_montague_days', 'Max Montague Days'),
    ('nyc_diversion_sat_ratio', 'Diversion Satisfaction'),
]:
    print(f"\n  {metric_label}:")
    print(f"  {'Severity Bin':<20} {'Baseline':>12} {'Mixed Future':>12} {'Wet Future':>12}")
    print(f"  {'-'*60}")

    for i, label in enumerate(severity_labels):
        lo, hi = severity_bins[i], severity_bins[i+1]
        vals = []
        for did in DATASETS:
            df = events[did]
            mask = (df['severity'] >= lo) & (df['severity'] < hi)
            v = df.loc[mask, metric]
            if len(v) > 0:
                vals.append(f"{v.median():.2f} (n={len(v)})")
            else:
                vals.append("N/A")
        print(f"  {label:<20} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}")

# ============================================================================
# TEST 3: Contribution amplification
# ============================================================================
print("\n" + "=" * 80)
print("TEST 3: Contribution Metrics by Zone")
print("=" * 80)

for did in DATASETS:
    cm = load_contribution_metrics(did)
    print(f"\n{LABELS[did]}:")
    for zone in [3, 4, 5, 6]:
        mask = cm['annual_max_zone'] == zone
        n = mask.sum()
        if n > 0:
            ratio = cm.loc[mask, 'contribution_ratio_180d'].dropna()
            storage = cm.loc[mask, 'annual_min_storage_pct'].dropna()
            print(f"  Zone {zone}: n={n:>4}, contrib_ratio_180d median={ratio.median():.1f}%, min_storage median={storage.median():.1f}%")

# ============================================================================
# TEST 4: Zone probability timing
# ============================================================================
print("\n" + "=" * 80)
print("TEST 4: Zone Probability Timing")
print("=" * 80)

zone_probs = {}
for did in DATASETS:
    zone_probs[did] = load_zone_probs(did)

# Compute P(drought zone) = P(zone 4) + P(zone 5) + P(zone 6) by week
print("\nP(drought zone >= 4) by month:")
print(f"{'Month':>6} {'Week':>5} | {'Baseline':>10} {'Mixed':>10} {'Wet':>10} | {'d_Mixed':>10} {'d_Wet':>10}")
print("-" * 70)

for wk, mo in zip(key_weeks, month_names):
    if wk in zone_probs['stationary_ensemble'].index:
        p_drought = []
        for did in DATASETS:
            zp = zone_probs[did].loc[wk]
            p_drought.append(zp['zone_4'] + zp['zone_5'] + zp['zone_6'])
        delta_low = p_drought[1] - p_drought[0]
        delta_high = p_drought[2] - p_drought[0]
        print(f"{mo:>4} {wk:>5} | {p_drought[0]:>9.1f}% {p_drought[1]:>9.1f}% {p_drought[2]:>9.1f}% | {delta_low:>+9.1f}% {delta_high:>+9.1f}%")

# ============================================================================
# TEST 5: June-to-September drawdown
# ============================================================================
print("\n" + "=" * 80)
print("TEST 5: June-September Storage Drawdown")
print("=" * 80)

for did in DATASETS:
    am = load_annual_metrics(did)
    am_all = am[am['period'] == 'all']
    june = am_all['june1_storage_pct'].dropna()
    sept = am_all['sept1_storage_pct'].dropna()
    drawdown = june - sept
    print(f"\n{LABELS[did]}:")
    print(f"  June 1 storage: p5={june.quantile(0.05):.1f}%, p50={june.median():.1f}%, p95={june.quantile(0.95):.1f}%")
    print(f"  Sept 1 storage: p5={sept.quantile(0.05):.1f}%, p50={sept.median():.1f}%, p95={sept.quantile(0.95):.1f}%")
    print(f"  Drawdown (June->Sept): p5={drawdown.quantile(0.05):.1f}%, p50={drawdown.median():.1f}%, p95={drawdown.quantile(0.95):.1f}%")

# ============================================================================
# TEST 6: Classification distribution per severity bin
# ============================================================================
print("\n" + "=" * 80)
print("TEST 6: Satisficing Classification by Severity Bin")
print("=" * 80)

for did in DATASETS:
    df = events[did]
    print(f"\n{LABELS[did]}:")
    for i, label in enumerate(severity_labels):
        lo, hi = severity_bins[i], severity_bins[i+1]
        mask = (df['severity'] >= lo) & (df['severity'] < hi)
        subset = df.loc[mask]
        if len(subset) > 0:
            class_counts = subset['classification'].value_counts(normalize=True) * 100
            print(f"  {label}: n={len(subset)}")
            for cls in ['pass', 'storage_fail', 'montague_fail', 'both_fail']:
                pct = class_counts.get(cls, 0)
                if pct > 0:
                    print(f"    {cls}: {pct:.1f}%")

# ============================================================================
# TEST 7: Event start month analysis
# ============================================================================
print("\n" + "=" * 80)
print("TEST 7: Drought Start Month vs Outcomes")
print("=" * 80)

for did in DATASETS:
    df = events[did]
    # Group by start_month bins (summer vs fall vs winter/spring)
    season_bins = {
        'Summer (Jun-Aug)': [6, 7, 8],
        'Fall (Sep-Nov)': [9, 10, 11],
        'Winter/Spring': [12, 1, 2, 3, 4, 5]
    }
    print(f"\n{LABELS[did]}:")
    for season, months in season_bins.items():
        mask = df['start_month'].isin(months)
        subset = df.loc[mask]
        if len(subset) > 0:
            print(f"  {season}: n={len(subset)}, severity={subset['severity'].median():.2f}, "
                  f"min_storage={subset['event_min_storage_pct'].median():.1f}%, "
                  f"contrib={subset['total_nyc_contribution_mg'].median():.0f} MG")

print("\n" + "=" * 80)
print("DIAGNOSTIC VALIDATION COMPLETE")
print("=" * 80)
