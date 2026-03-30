"""
Reusable helpers for severity × magnitude heatmap figures.

Grid computation functions operate on the per-event-metrics DataFrames
produced by ``08_calculate_event_metrics.py`` (loaded via
``methods.load.load_event_metrics``).
"""

import numpy as np


# ── default constants (importable, overridable) ─────────────────────
N_HEAT_BINS = 10
MIN_COUNT = 5
SEV_MAX = 4.0
MAG_MAX = 50.0
WORST_STORAGE_THRESH = 10.0
SATISFICING_THRESHOLD = 0.90


def make_shared_edges(all_data, datasets, n_bins=N_HEAT_BINS,
                      sev_max=SEV_MAX, mag_max=MAG_MAX):
    """Compute shared severity/magnitude bin edges across datasets.

    Parameters
    ----------
    all_data : dict
        ``{dataset_id: DataFrame}`` with ``severity`` and ``magnitude`` columns.
    datasets : list of str
        Dataset IDs to pool.
    n_bins : int
        Number of bins per axis.
    sev_max, mag_max : float
        Upper axis caps.

    Returns
    -------
    sev_edges, mag_edges, sev_centers, mag_centers : np.ndarray
    """
    all_sev = np.concatenate([all_data[d]['severity'].values for d in datasets])
    all_mag = np.concatenate([all_data[d]['magnitude'].values for d in datasets])
    sev_edges = np.linspace(all_sev.min(), sev_max, n_bins + 1)
    mag_edges = np.linspace(all_mag.min(), mag_max, n_bins + 1)
    sev_centers = 0.5 * (sev_edges[:-1] + sev_edges[1:])
    mag_centers = 0.5 * (mag_edges[:-1] + mag_edges[1:])
    return sev_edges, mag_edges, sev_centers, mag_centers


def compute_min_storage_grid(df, sev_edges, mag_edges, min_count=MIN_COUNT):
    """2-D grid of worst-case (absolute minimum) storage per bin.

    Returns
    -------
    min_grid : np.ndarray (ns × nm), NaN where count < min_count
    count_grid : np.ndarray (ns × nm)
    """
    sev = df['severity'].values
    mag = df['magnitude'].values
    sto = df['event_min_storage_pct'].values

    sev_idx = np.digitize(sev, sev_edges) - 1
    mag_idx = np.digitize(mag, mag_edges) - 1

    ns = len(sev_edges) - 1
    nm = len(mag_edges) - 1

    min_grid = np.full((ns, nm), np.nan)
    count_grid = np.zeros((ns, nm), dtype=int)

    for i in range(ns):
        for j in range(nm):
            mask = (sev_idx == i) & (mag_idx == j)
            cnt = mask.sum()
            count_grid[i, j] = cnt
            if cnt < min_count:
                continue
            min_grid[i, j] = sto[mask].min()

    return min_grid, count_grid


def compute_emergency_grid(df, sev_edges, mag_edges, min_count=MIN_COUNT):
    """2-D grid of fraction of events avoiding Drought Emergency per bin.

    Returns
    -------
    frac_grid : np.ndarray (ns × nm), NaN where count < min_count
    count_grid : np.ndarray (ns × nm)
    """
    sev = df['severity'].values
    mag = df['magnitude'].values
    zone = df['ffmp_zone_at_min'].values

    sev_idx = np.digitize(sev, sev_edges) - 1
    mag_idx = np.digitize(mag, mag_edges) - 1

    ns = len(sev_edges) - 1
    nm = len(mag_edges) - 1

    frac_grid = np.full((ns, nm), np.nan)
    count_grid = np.zeros((ns, nm), dtype=int)

    for i in range(ns):
        for j in range(nm):
            mask = (sev_idx == i) & (mag_idx == j)
            cnt = mask.sum()
            count_grid[i, j] = cnt
            if cnt < min_count:
                continue
            n_above = (zone[mask] != 'Emergency').sum()
            frac_grid[i, j] = n_above / cnt

    return frac_grid, count_grid
