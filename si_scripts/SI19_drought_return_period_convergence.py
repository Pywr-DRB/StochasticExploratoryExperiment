"""
SI19: Convergence of extreme drought joint-exceedance return period.

Measures how the joint-exceedance return period T_W (the same quantity
shown in Fig9 panel (a)) stabilizes as the ensemble grows, evaluated
at three (severity, magnitude) thresholds that span the focal-region
neighborhood.

Return period definition (reused from Fig9 / methods.return_period):

    T_W(s*, m*) = T_R − E[D | sev ≥ s* AND mag ≥ m*]
    T_R         = E[L_interarrival] / P_exc(s*, m*)
    P_exc       = 1 − F_S(s*) − F_M(m*) + Ĉ(F_S(s*), F_M(m*))

i.e. the duration-adjusted drought-free interval at the lower-left
corner of a heatmap cell, with the joint exceedance probability
estimated via the empirical copula (Deheuvels 1979; Nelsen 2006 §2.4;
Salvadori & De Michele 2004). Computed by
``methods.return_period.compute_return_period_grid_exceedance`` and
read at the corner cell whose lower-left corner is the chosen
(severity, magnitude) threshold.

Convergence experiment:
  - subset sizes N log-spaced from 1 to N_realizations,
  - 200 bootstrap subsamples per N (without replacement),
  - for each subsample, subset the event table to those realization_ids
    and call the Fig9 return-period function with bin edges placing
    each (s*, m*) at a cell corner,
  - report 5/50/95 bootstrap percentiles of T_W vs. N for each threshold.

Three thresholds (lower-left corners of severity-magnitude cells):
  1. (sev ≥ 2.0, mag ≥ 5)   — moderate severe drought
  2. (sev ≥ 2.5, mag ≥ 15)  — focal-region center, near 1960s drought
  3. (sev ≥ 3.0, mag ≥ 30)  — deep tail

Usage:
    python SI19_drought_return_period_convergence.py [dataset_id] [ssi_window]

Defaults: dataset_id = stationary_ensemble, ssi_window = 3
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from methods.config import (
    FIG_DIR, N_YEARS, DATASET_CONFIGS, verify_dataset_id,
)
from methods.load import load_event_metrics, load_drought_events
from methods.return_period import compute_return_period_grid_exceedance
from methods.plotting.styles import (
    DATASET_COLORS, DATASET_LABELS,
    ALPHA_FILL, LINEWIDTH_MEDIUM, DPI_PRINT,
    apply_publication_style,
)

FIG_OUTPUT_DIR = f"{FIG_DIR}/SI19_drought_rp_convergence"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

# (label, severity_threshold, magnitude_threshold)
# Each pair is the lower-left corner of the cell whose T_W will be tracked.
# Spans the focal region of Fig9: moderate → focal-center → deep tail.
THRESHOLDS = [
    ("Moderate (sev>=2.0, mag>=5)",   2.0,  5.0),
    ("Severe (sev>=2.5, mag>=15)",    2.5, 15.0),
    ("Extreme (sev>=3.0, mag>=30)",   3.0, 30.0),
]

# 1960s drought-of-record reference. Added as a 4th convergence panel
# only when the script is run against the stationary baseline (matches
# Fig9, where the DoR triangle is drawn only on panel (a)). The (sev, mag)
# values are read from the observed-events table at this date rather than
# hardcoded — same convention as Fig9 (DOR_TARGET_DATE).
DOR_TARGET_DATE = pd.Timestamp('1964-12-01')
DOR_DATASET_ID = 'stationary_ensemble'

# Bootstrap settings
N_BOOTSTRAP_SAMPLES = 200
N_SUBSET_STEPS = 30
RNG_SEED = 42

# min_count for the return-period calc. Fig9 uses 5 to mask sparsely-
# populated cells visually. For the convergence experiment we want to
# observe the *point estimate* stabilising, so allow any non-empty
# exceedance set and separately track when ≥5 exceedances are reached.
MIN_COUNT_CONVERGENCE = 1
MIN_COUNT_RELIABLE = 5


def _resolve_1960s_dor_threshold(dataset_id, ssi_window):
    """Return ("1960s DoR (sev=…, mag=…)", sev, mag) or None.

    Reads observed drought events for *dataset_id* and picks the event
    active on DOR_TARGET_DATE — same logic as Fig9's red triangle. The
    threshold is the observed (severity, magnitude) of that event, so
    the T_W convergence panel answers: "how long does it take to pin
    down the return period of an event AT LEAST as severe and as long
    as the 1960s drought of record?"
    """
    try:
        obs = load_drought_events(dataset_id, ssi_window, observed=True)
    except FileNotFoundError as e:
        print(f"  WARN: observed drought events not found, skipping DoR panel: {e}")
        return None
    obs_start = pd.to_datetime(obs['start'])
    obs_end = pd.to_datetime(obs['end'])
    mask = (obs_start <= DOR_TARGET_DATE) & (obs_end >= DOR_TARGET_DATE)
    matches = obs[mask.values]
    if matches.empty:
        print(f"  WARN: no observed drought spans {DOR_TARGET_DATE.date()}; "
              "skipping DoR panel")
        return None
    row = matches.iloc[0]
    dor_sev = float(abs(row['severity']))
    dor_mag = float(abs(row['magnitude']))
    label = f"1960s drought of record (sev>={dor_sev:.2f}, mag>={dor_mag:.1f})"
    return label, dor_sev, dor_mag


def _eval_T_W_at_threshold(df, sev_thresh, mag_thresh, n_years,
                            min_count=MIN_COUNT_CONVERGENCE):
    """Return (T_W in years, exceedance count) at one (sev, mag) corner.

    Builds 2-edge severity/magnitude grids so the requested threshold is
    the lower-left corner of a single cell, then calls the Fig9 return-
    period function and reads cell (0, 0).
    """
    sev_max = max(float(df['severity'].max()), sev_thresh) + 1.0
    mag_max = max(float(df['magnitude'].max()), mag_thresh) + 1.0
    sev_edges = np.array([sev_thresh, sev_max])
    mag_edges = np.array([mag_thresh, mag_max])

    _T_R, _dur, T_W, _cnt = compute_return_period_grid_exceedance(
        df, sev_edges, mag_edges, n_years, min_count=min_count,
    )
    n_exc = int(
        ((df['severity'].values >= sev_thresh)
         & (df['magnitude'].values >= mag_thresh)).sum()
    )
    return float(T_W[0, 0]), n_exc


def compute_convergence_bands(df, realization_ids, ssi_window, thresholds):
    """Bootstrap T_W vs. subset size N at each (sev, mag) threshold.

    Returns
    -------
    n_subset_sizes : np.ndarray
    bands : np.ndarray, shape (n_subset_sizes, 3, n_thresholds)
        Percentiles [5, 50, 95] of T_W across bootstrap samples.
    reliable_frac : np.ndarray, shape (n_subset_sizes, n_thresholds)
        Fraction of bootstrap samples reaching ≥MIN_COUNT_RELIABLE
        exceedance events at each threshold.
    """
    n_realizations = len(realization_ids)
    n_thresh = len(thresholds)

    n_subset_sizes = np.unique(
        np.linspace(1, n_realizations, N_SUBSET_STEPS).round().astype(int)
    )

    bands = np.full((len(n_subset_sizes), 3, n_thresh), np.nan)
    reliable_frac = np.zeros((len(n_subset_sizes), n_thresh))

    rng = np.random.default_rng(RNG_SEED)
    ids_arr = np.asarray(realization_ids)
    rid_groups = df.groupby('realization_id').indices

    for i, n in enumerate(n_subset_sizes):
        boot_T_W = np.full((N_BOOTSTRAP_SAMPLES, n_thresh), np.nan)
        boot_reliable = np.zeros((N_BOOTSTRAP_SAMPLES, n_thresh), dtype=bool)
        for b in range(N_BOOTSTRAP_SAMPLES):
            sampled_ids = rng.choice(ids_arr, size=n, replace=False)
            sub_idx = np.concatenate(
                [rid_groups[rid] for rid in sampled_ids if rid in rid_groups]
            ) if any(rid in rid_groups for rid in sampled_ids) else np.array([], dtype=int)
            if sub_idx.size == 0:
                continue
            sub_df = df.iloc[sub_idx]
            # Each call subsets to events in this bootstrap's realizations,
            # then reads T_W at each threshold corner. n_years stays at the
            # per-realization N_YEARS — total ensemble years is n × N_YEARS.
            for k, (_lbl, s_thr, m_thr) in enumerate(thresholds):
                T_W_val, n_exc = _eval_T_W_at_threshold(
                    sub_df, s_thr, m_thr, N_YEARS,
                    min_count=MIN_COUNT_CONVERGENCE,
                )
                boot_T_W[b, k] = T_W_val
                boot_reliable[b, k] = (n_exc >= MIN_COUNT_RELIABLE)

        for k in range(n_thresh):
            col = boot_T_W[:, k]
            valid = col[np.isfinite(col)]
            if valid.size:
                bands[i, :, k] = np.percentile(valid, [5, 50, 95])
            reliable_frac[i, k] = boot_reliable[:, k].mean()

    return n_subset_sizes, bands, reliable_frac


def plot_convergence(n_subset_sizes, bands, reliable_frac,
                     dataset_id, ssi_window, thresholds, fname):
    apply_publication_style()
    color = DATASET_COLORS.get(dataset_id, '#1f77b4')
    n_thresh = len(thresholds)
    n_realizations = int(n_subset_sizes.max())

    fig, axes = plt.subplots(1, n_thresh, figsize=(5.0 * n_thresh + 0.8, 5.5),
                             squeeze=False)
    axes = axes[0]

    for k, (label, s_thr, m_thr) in enumerate(thresholds):
        ax = axes[k]
        lo = bands[:, 0, k]
        med = bands[:, 1, k]
        hi = bands[:, 2, k]

        valid = np.isfinite(med)
        ax.fill_between(
            n_subset_sizes[valid], lo[valid], hi[valid],
            alpha=ALPHA_FILL, color=color,
            label=f'Bootstrap 5–95% ({N_BOOTSTRAP_SAMPLES} subsamples)',
        )
        ax.plot(
            n_subset_sizes[valid], med[valid],
            color=color, linewidth=LINEWIDTH_MEDIUM,
            label='Bootstrap median of T_W',
        )

        # Mark the smallest N at which ≥50% of bootstrap samples reach
        # MIN_COUNT_RELIABLE exceedance events (the Fig9 reliability bar).
        reliable_mask = reliable_frac[:, k] >= 0.5
        if reliable_mask.any():
            n_reliable = int(n_subset_sizes[reliable_mask].min())
            ax.axvline(
                n_reliable, color='#555555', linestyle=':',
                linewidth=1.2,
                label=(f'N where ≥50% of subsamples\n'
                       f'have ≥{MIN_COUNT_RELIABLE} exceedance events '
                       f'(N={n_reliable})'),
            )

        ax.set_yscale('log')
        ax.set_xlim(0, n_realizations)
        ax.set_xlabel('Number of realizations N\n'
                      '(subsampled without replacement)')
        ax.set_ylabel('Joint-exceedance return period T_W (years)')
        ax.set_title(label)
        ax.legend(loc='best', frameon=True, fontsize='small')
        ax.grid(False)

        # Top secondary axis: ensemble years = N × N_YEARS.
        sec = ax.secondary_xaxis(
            'top',
            functions=(lambda x: x * N_YEARS, lambda y: y / N_YEARS),
        )
        sec.set_xlabel(f'Ensemble years (N × {N_YEARS})')

    ds_label = DATASET_LABELS.get(dataset_id, dataset_id)
    fig.suptitle(
        f'Drought return-period convergence — {ds_label}, SSI-{ssi_window}\n'
        f'(joint-exceedance T_W via empirical copula; same metric as Fig9 panel a)',
        fontsize='medium', y=1.0,
    )
    footer = (
        'Each panel evaluates the joint-exceedance return period T_W at one (severity, magnitude) '
        'lower-left-corner threshold, using the empirical-copula formula reused from Fig9 '
        '(methods.return_period.compute_return_period_grid_exceedance). For each subset size N, '
        f'{N_BOOTSTRAP_SAMPLES} bootstrap subsamples are drawn without replacement; the band shows '
        'the 5–95% range and the line the median of T_W across those subsamples. Convergence is '
        'reached when both the band tightens and the median flattens. The dotted vertical line '
        f'marks the smallest N at which ≥50% of subsamples contain ≥{MIN_COUNT_RELIABLE} '
        'exceedance events (the Fig9 reliability bar). T_W axis is log-scaled.'
    )
    fig.text(0.5, -0.05, footer, ha='center', va='top',
             fontsize='small', wrap=True)

    plt.tight_layout()
    plt.savefig(fname, dpi=DPI_PRINT, bbox_inches='tight')
    print(f"Saved: {fname}")
    plt.close(fig)


def run(dataset_id, ssi_window):
    verify_dataset_id(dataset_id)
    print(f"SI19: drought return-period convergence — "
          f"{dataset_id}, SSI-{ssi_window}")

    df = load_event_metrics(dataset_id, ssi_window)
    realization_ids = sorted(df['realization_id'].unique().tolist())
    print(f"  Events: {len(df):,}; realizations with ≥1 event: "
          f"{len(realization_ids):,}")

    # Stationary baseline gets an extra 1960s-DoR convergence panel —
    # the climate-adjusted scenarios share the historical record but
    # Fig9 panel (a) is the only place the DoR triangle is drawn, so
    # the convergence reference is restricted to the same dataset.
    thresholds = list(THRESHOLDS)
    if dataset_id == DOR_DATASET_ID:
        dor_threshold = _resolve_1960s_dor_threshold(dataset_id, ssi_window)
        if dor_threshold is not None:
            thresholds.append(dor_threshold)

    # Quick sanity check on threshold population.
    for label, s_thr, m_thr in thresholds:
        n_exc = int(((df['severity'] >= s_thr)
                     & (df['magnitude'] >= m_thr)).sum())
        print(f"    {label}: {n_exc:,} exceedance events in full ensemble")

    n_subset_sizes, bands, reliable_frac = compute_convergence_bands(
        df, realization_ids, ssi_window, thresholds,
    )

    fname = (f"{FIG_OUTPUT_DIR}/SI19_drought_rp_convergence_"
             f"{dataset_id}_ssi{ssi_window}.png")
    plot_convergence(n_subset_sizes, bands, reliable_frac,
                     dataset_id, ssi_window, thresholds, fname)


def main():
    dataset_id = sys.argv[1] if len(sys.argv) > 1 else 'stationary_ensemble'
    ssi_window = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    run(dataset_id, ssi_window)
    print("Done.")


if __name__ == '__main__':
    main()
