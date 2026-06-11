"""
Validate candidate drought-DYNAMICS metrics BEFORE committing to any rerun.

All candidates are derived from the existing drought_events CSVs (SSI-based
event definitions), which already carry: duration (months), severity (min SSI),
magnitude (sum SSI), avg_severity, max_severity_date, recovery_period (months
peak->end), prior_{1,3,6}m_surplus. No pipeline rerun is needed to compute or
test them.

For each candidate metric we report: definition, units, distribution,
degenerate/edge-case fraction, and Pearson correlation with the existing hazard
features (to confirm it adds an INDEPENDENT axis rather than restating size).

Run directly (lightweight; reads CSVs only). Prints a report to stdout.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from methods.config import DROUGHT_METRICS_DIR, DATASET_CONFIGS

SSI_WINDOW = 3
DATASETS = list(DATASET_CONFIGS.keys())
MIN_DURATION_MONTHS = 1   # keep all; report sensitivity separately


def load_events():
    frames = []
    for ds in DATASETS:
        f = os.path.join(DROUGHT_METRICS_DIR,
                         f"{ds}_ssi{SSI_WINDOW}_drought_events.csv")
        df = pd.read_csv(f, parse_dates=['start', 'end', 'max_severity_date'])
        df['dataset'] = ds
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def derive(df):
    d = df.copy()
    # absolute (positive) intensity conventions
    d['sev_abs'] = d['severity'].abs()
    d['avgsev_abs'] = d['avg_severity'].abs()
    d['mag_abs'] = d['magnitude'].abs()

    # time-to-peak (months) = duration - recovery_period  (both in months)
    d['time_to_peak_m'] = d['duration'] - d['recovery_period']
    d['time_to_peak_frac'] = d['time_to_peak_m'] / d['duration']

    # intensification (onset) and recovery rates: |SSI| per month
    # floor denominators at 0.5 month to avoid divide-by-zero for peak-at-edge
    d['onset_rate'] = d['sev_abs'] / d['time_to_peak_m'].clip(lower=0.5)
    d['recovery_rate'] = d['sev_abs'] / d['recovery_period'].clip(lower=0.5)

    # peakedness (peak vs mean intensity)
    d['peakedness'] = d['sev_abs'] / d['avgsev_abs'].replace(0, np.nan)

    # seasonality (cyclic) of onset and of peak
    d['start_month'] = d['start'].dt.month
    d['peak_month'] = d['max_severity_date'].dt.month
    for col, m in [('onset', 'start_month'), ('peak', 'peak_month')]:
        d[f'{col}_sin'] = np.sin(2 * np.pi * d[m] / 12.0)
        d[f'{col}_cos'] = np.cos(2 * np.pi * d[m] / 12.0)

    # summer exposure: fraction of event months in Jun-Sep (high-demand season)
    def summer_frac(row):
        rng = pd.period_range(row['start'], row['end'], freq='M')
        if len(rng) == 0:
            return np.nan
        return np.mean([(p.month in (6, 7, 8, 9)) for p in rng])
    # vectorized-ish: compute on unique (start,end) is overkill; sample-safe loop
    d['summer_frac'] = d.apply(summer_frac, axis=1)

    # antecedent wetness (already computed upstream) — separate CATEGORY
    # (kept here only to report; NOT a hazard feature)
    return d


def describe(series, name, unit):
    s = pd.to_numeric(series, errors='coerce')
    finite = s.replace([np.inf, -np.inf], np.nan).dropna()
    n_nan = int(s.isna().sum() + np.isinf(s.replace(np.nan, 0)).sum())
    q = finite.quantile([0, .05, .25, .5, .75, .95, 1.0])
    print(f"\n  [{name}]  ({unit})")
    print(f"    n={len(finite):,}  NaN/inf={n_nan:,}  "
          f"mean={finite.mean():.3f}  std={finite.std():.3f}")
    print(f"    min={q[0]:.3f}  p5={q[.05]:.3f}  p25={q[.25]:.3f}  "
          f"med={q[.5]:.3f}  p75={q[.75]:.3f}  p95={q[.95]:.3f}  "
          f"max={q[1.0]:.3f}")


def main():
    print("#" * 76)
    print("# DROUGHT-DYNAMICS METRIC ASSESSMENT (SSI-3, derived from "
          "drought_events)")
    print("#" * 76)
    raw = load_events()
    print(f"\nLoaded {len(raw):,} SSI-3 drought events "
          f"({', '.join(f'{ds}:{(raw.dataset==ds).sum():,}' for ds in DATASETS)})")
    print("NOTE: duration & recovery_period are in MONTHS (SSI is monthly).")

    # Sample for tractable assessment (distributions/correlations are stable);
    # the expensive summer_frac loop makes full-set derivation slow.
    if len(raw) > 40000:
        raw = raw.sample(40000, random_state=0).reset_index(drop=True)
        print(f"Assessing on a random sample of {len(raw):,} events.")

    d = derive(raw)

    # ---- edge cases / validity flags ----
    print("\n" + "=" * 70)
    print("EDGE-CASE / VALIDITY CHECKS")
    print("=" * 70)
    n = len(d)
    print(f"  duration == 1 month            : {(d['duration']==1).sum():,} "
          f"({100*(d['duration']==1).mean():.1f}%)")
    print(f"  time_to_peak_m == 0 (peak@start): {(d['time_to_peak_m']==0).sum():,} "
          f"({100*(d['time_to_peak_m']==0).mean():.1f}%)  "
          f"-> onset_rate uses 0.5-month floor")
    print(f"  time_to_peak_m < 0 (INVALID)   : {(d['time_to_peak_m']<0).sum():,}")
    print(f"  recovery_period == 0           : {(d['recovery_period']==0).sum():,}")
    print(f"  max_severity_date NaT          : {d['max_severity_date'].isna().sum():,}")
    print(f"  peakedness NaN (avgsev=0)       : {d['peakedness'].isna().sum():,}")

    # sensitivity to a 30-day (~1 month) minimum-duration filter
    for thr in (1, 2, 3):
        keep = (d['duration'] >= thr).mean()
        print(f"  fraction retained at duration >= {thr} mo: {100*keep:.1f}%")

    # ---- distributions ----
    print("\n" + "=" * 70)
    print("CANDIDATE METRIC DISTRIBUTIONS")
    print("=" * 70)
    describe(d['duration'], 'duration', 'months')
    describe(d['sev_abs'], 'severity |min SSI|', 'SSI')
    describe(d['time_to_peak_m'], 'time_to_peak', 'months')
    describe(d['time_to_peak_frac'], 'time_to_peak_frac', '0-1')
    describe(d['onset_rate'], 'onset_rate (intensification)', '|SSI|/month')
    describe(d['recovery_rate'], 'recovery_rate', '|SSI|/month')
    describe(d['recovery_period'], 'recovery_period', 'months')
    describe(d['peakedness'], 'peakedness', 'ratio')
    describe(d['summer_frac'], 'summer_frac (Jun-Sep)', '0-1')
    describe(d['prior_3m_surplus'], 'prior_3m_surplus (antecedent)', 'SSI-sum')

    # ---- independence from existing hazard features ----
    print("\n" + "=" * 70)
    print("CORRELATION OF NEW METRICS WITH EXISTING HAZARD FEATURES")
    print("(low |r| => adds an independent axis; high |r| => redundant)")
    print("=" * 70)
    existing = ['duration', 'sev_abs', 'mag_abs', 'avgsev_abs']
    new = ['time_to_peak_m', 'time_to_peak_frac', 'onset_rate',
           'recovery_rate', 'recovery_period', 'peakedness',
           'onset_sin', 'onset_cos', 'summer_frac']
    sub = d[existing + new].replace([np.inf, -np.inf], np.nan).dropna()
    corr = sub.corr()
    block = corr.loc[new, existing]
    with pd.option_context('display.float_format', '{:+.2f}'.format,
                           'display.width', 200):
        print(block.to_string())

    # also: correlation AMONG the new dynamics features
    print("\n  Correlation among new dynamics features:")
    cn = corr.loc[new, new]
    with pd.option_context('display.float_format', '{:+.2f}'.format,
                           'display.width', 200):
        print(cn.to_string())

    print("\nDONE (assessment only; nothing written, no rerun).")


if __name__ == '__main__':
    main()
