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
SEV_MIN = 1.0
SEV_MAX = 4.5
MAG_MIN = 1.0
MAG_MAX = 100.0
WORST_STORAGE_THRESH = 10.0
SATISFICING_THRESHOLD = 0.90

# ── Shared grid configuration for Fig9 / Fig10 ──────────────────────
# Central location so heatmap and dynamics figures use identical bins.
GRID_N_BINS = 16
GRID_LOG_MAG = True
GRID_TARGET_SEV_BIN = 7   # 0-indexed severity bin for focal cell
GRID_TARGET_MAG_BIN = 11   # 0-indexed magnitude bin for focal cell


def make_shared_edges(all_data, datasets, n_bins=N_HEAT_BINS,
                      sev_min=SEV_MIN, sev_max=SEV_MAX,
                      mag_min=MAG_MIN, mag_max=MAG_MAX):
    """Compute shared severity/magnitude bin edges across datasets.

    Parameters
    ----------
    all_data : dict
        ``{dataset_id: DataFrame}`` with ``severity`` and ``magnitude`` columns.
    datasets : list of str
        Dataset IDs to pool.
    n_bins : int
        Number of bins per axis.
    sev_min, sev_max, mag_min, mag_max : float
        Axis bounds.

    Returns
    -------
    sev_edges, mag_edges, sev_centers, mag_centers : np.ndarray
    """
    sev_edges = np.linspace(sev_min, sev_max, n_bins + 1)
    mag_edges = np.linspace(mag_min, mag_max, n_bins + 1)
    sev_centers = 0.5 * (sev_edges[:-1] + sev_edges[1:])
    mag_centers = 0.5 * (mag_edges[:-1] + mag_edges[1:])
    return sev_edges, mag_edges, sev_centers, mag_centers


def make_shared_edges_logmag(all_data, datasets, n_bins=GRID_N_BINS,
                             sev_min=SEV_MIN, sev_max=SEV_MAX,
                             mag_min=MAG_MIN, mag_max=MAG_MAX):
    """Like :func:`make_shared_edges` but with log-spaced magnitude bins.

    Returns
    -------
    sev_edges, mag_edges, sev_centers, mag_centers : np.ndarray
    """
    sev_edges = np.linspace(sev_min, sev_max, n_bins + 1)
    sev_centers = 0.5 * (sev_edges[:-1] + sev_edges[1:])
    mag_edges = np.logspace(np.log10(mag_min), np.log10(mag_max), n_bins + 1)
    mag_centers = np.sqrt(mag_edges[:-1] * mag_edges[1:])
    return sev_edges, mag_edges, sev_centers, mag_centers


def assign_grid_bins(df, sev_edges, mag_edges):
    """Assign severity/magnitude grid-cell indices to each event row.

    Adds columns ``sev_bin`` and ``mag_bin`` (0-based indices into the grid)
    to the DataFrame. Events outside the grid edges get index -1 or n_bins
    (clipped to valid range or dropped by callers).

    Parameters
    ----------
    df : pd.DataFrame
        Must have ``severity`` and ``magnitude`` columns.
    sev_edges, mag_edges : np.ndarray
        Bin edges from :func:`make_shared_edges`.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ``sev_bin`` and ``mag_bin`` columns added.
    """
    out = df.copy()
    ns = len(sev_edges) - 1
    nm = len(mag_edges) - 1
    out['sev_bin'] = np.clip(np.digitize(out['severity'].values, sev_edges) - 1, 0, ns - 1)
    out['mag_bin'] = np.clip(np.digitize(out['magnitude'].values, mag_edges) - 1, 0, nm - 1)
    return out


def select_from_grid_cell(df_binned, sev_bin, mag_bin,
                          rank_col='event_min_storage_pct',
                          ascending=True, n=1):
    """Select events from a specific grid cell, ranked by a metric.

    Parameters
    ----------
    df_binned : pd.DataFrame
        Output of :func:`assign_grid_bins`.
    sev_bin, mag_bin : int
        Grid-cell indices.
    rank_col : str
        Column to rank by (default: worst storage → ascending=True).
    ascending : bool
        If True, the smallest values rank first (worst-case selection).
    n : int
        Number of events to return from this cell.

    Returns
    -------
    pd.DataFrame
        Up to *n* rows from the requested cell, sorted by *rank_col*.
    """
    cell = df_binned[(df_binned['sev_bin'] == sev_bin) &
                     (df_binned['mag_bin'] == mag_bin)]
    if len(cell) == 0:
        return cell
    return cell.sort_values(rank_col, ascending=ascending).head(n)


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


def compute_exceedance_rate_grid(df, sev_edges, mag_edges, n_years,
                                 min_count=MIN_COUNT):
    """2-D grid of empirical exceedance rate (events/year) per bin.

    For each bin, rate = count / (n_realizations * n_years).

    Parameters
    ----------
    df : pd.DataFrame
        Event metrics with ``severity``, ``magnitude``, ``realization_id``.
    sev_edges, mag_edges : np.ndarray
    n_years : int
        Simulation years per realization.
    min_count : int
        Bins with fewer events are NaN.

    Returns
    -------
    rate_grid : np.ndarray (ns x nm), NaN where count < min_count
    count_grid : np.ndarray (ns x nm)
    """
    sev = df['severity'].values
    mag = df['magnitude'].values
    n_realizations = df['realization_id'].nunique()
    total_years = n_realizations * n_years

    sev_idx = np.digitize(sev, sev_edges) - 1
    mag_idx = np.digitize(mag, mag_edges) - 1

    ns = len(sev_edges) - 1
    nm = len(mag_edges) - 1

    rate_grid = np.full((ns, nm), np.nan)
    count_grid = np.zeros((ns, nm), dtype=int)

    for i in range(ns):
        for j in range(nm):
            mask = (sev_idx == i) & (mag_idx == j)
            cnt = mask.sum()
            count_grid[i, j] = cnt
            if cnt < min_count:
                continue
            rate_grid[i, j] = cnt / total_years

    return rate_grid, count_grid


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
