"""
Reusable helpers for severity × magnitude heatmap figures.

Grid computation functions operate on the per-event-metrics DataFrames
produced by ``08_calculate_event_metrics.py`` (loaded via
``methods.load.load_event_metrics``).

Shared figure / focal-region specifications are defined once in
``methods.config`` and re-exported here so existing imports keep
working.
"""

import numpy as np

from methods.config import (
    GRID_N_BINS, GRID_LOG_MAG, GRID_TARGET_SEV_BIN, GRID_TARGET_MAG_BIN,
    SEV_MIN, SEV_MAX, MAG_MIN, MAG_MAX,
    MIN_COUNT_PER_BIN,
    FOCAL_FRAC_THRESH, FOCAL_RP_THRESH_YEARS, FOCAL_WORST_STORAGE_THRESH,
    WORST_STORAGE_THRESH, SATISFICING_THRESHOLD,
)

# Backwards-compatible name (a few callers still import the legacy alias)
N_HEAT_BINS = 10
MIN_COUNT = MIN_COUNT_PER_BIN


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


def draw_focal_boundary(ax, sev_edges, mag_edges, focal_cells,
                        edgecolor='white', linewidth=2.0, zorder=9,
                        halo=True, halo_color='#000000', halo_linewidth=3.5,
                        halo_zorder=8):
    """Draw a continuous boundary around the outer edge of focal-region cells.

    Instead of individual rectangles per cell, this traces the outer contour
    of the connected focal region by walking the boundary edges of the
    cell set on a rectilinear grid.

    When *halo* is True (default), a darker outline is drawn first beneath the
    main line so the boundary stays visible against bright and dark cell
    colours alike.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    sev_edges, mag_edges : np.ndarray
        Bin edges (length n_bins + 1).
    focal_cells : set of (i, j)
        Grid-cell indices in the focal region.
    edgecolor : str
        Colour of the inner (core) line.
    linewidth : float
        Width of the inner (core) line.
    zorder : int
        z-order of the inner (core) line.
    halo : bool
        If True, draw a darker outline of width *halo_linewidth* below the
        core line.
    halo_color : str
    halo_linewidth : float
    halo_zorder : int
    """
    if not focal_cells:
        return

    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    # Collect all boundary edge segments.
    # For each cell (i, j), check the 4 neighbours.  If a neighbour is NOT
    # in the focal set, the shared edge is a boundary segment.
    segments = []
    for (i, j) in focal_cells:
        x0, x1 = sev_edges[i], sev_edges[i + 1]
        y0, y1 = mag_edges[j], mag_edges[j + 1]

        # bottom edge: neighbour (i, j-1)
        if (i, j - 1) not in focal_cells:
            segments.append(((x0, y0), (x1, y0)))
        # top edge: neighbour (i, j+1)
        if (i, j + 1) not in focal_cells:
            segments.append(((x0, y1), (x1, y1)))
        # left edge: neighbour (i-1, j)
        if (i - 1, j) not in focal_cells:
            segments.append(((x0, y0), (x0, y1)))
        # right edge: neighbour (i+1, j)
        if (i + 1, j) not in focal_cells:
            segments.append(((x1, y0), (x1, y1)))

    # Chain segments into closed loops for clean rendering.
    # Build adjacency: at each vertex, which segments meet?
    from collections import defaultdict
    adjacency = defaultdict(list)
    for idx, (p0, p1) in enumerate(segments):
        adjacency[p0].append(idx)
        adjacency[p1].append(idx)

    used = [False] * len(segments)
    loops = []

    for start_idx in range(len(segments)):
        if used[start_idx]:
            continue
        used[start_idx] = True
        loop = [segments[start_idx][0], segments[start_idx][1]]

        while loop[-1] != loop[0]:
            current = loop[-1]
            found = False
            for seg_idx in adjacency[current]:
                if used[seg_idx]:
                    continue
                used[seg_idx] = True
                p0, p1 = segments[seg_idx]
                next_pt = p1 if p0 == current else p0
                loop.append(next_pt)
                found = True
                break
            if not found:
                break

        loops.append(loop)

    # Draw each loop as a closed path — halo first (if requested), then core.
    for loop in loops:
        verts = loop
        codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
        path = Path(verts, codes)
        if halo:
            ax.add_patch(PathPatch(path, facecolor='none', edgecolor=halo_color,
                                   linewidth=halo_linewidth, zorder=halo_zorder))
        ax.add_patch(PathPatch(path, facecolor='none', edgecolor=edgecolor,
                               linewidth=linewidth, zorder=zorder))


def identify_focal_region(T_W_grids, frac_grids, min_grids, datasets,
                          frac_thresh=FOCAL_FRAC_THRESH,
                          rp_thresh_years=FOCAL_RP_THRESH_YEARS,
                          storage_thresh=FOCAL_WORST_STORAGE_THRESH):
    """Identify grid cells meeting multi-metric focal-region criteria.

    Thresholding is on the duration-adjusted return period
    ``T_W = T_R - E[D|bin]`` (Bonaccorso-Shiau interarrival time minus
    mean event duration), so cells with long-duration drought events
    are evaluated on the *drought-free* waiting interval rather than
    the raw recurrence interval.

    Criteria
    --------
    1. Fraction avoiding emergency < *frac_thresh* in ALL datasets
    2. T_W ≤ *rp_thresh_years* in ALL datasets
    3. Worst-case storage < *storage_thresh* in at least 1 dataset

    Returns
    -------
    focal_cells : set of (i, j) tuples
    """
    ns, nm = T_W_grids[datasets[0]].shape
    focal_cells = set()

    for i in range(ns):
        for j in range(nm):
            if not all(
                not np.isnan(T_W_grids[d][i, j]) and
                T_W_grids[d][i, j] <= rp_thresh_years
                for d in datasets
            ):
                continue
            if not all(
                not np.isnan(frac_grids[d][i, j]) and
                frac_grids[d][i, j] < frac_thresh
                for d in datasets
            ):
                continue
            if not any(
                not np.isnan(min_grids[d][i, j]) and
                min_grids[d][i, j] < storage_thresh
                for d in datasets
            ):
                continue
            focal_cells.add((i, j))

    return focal_cells


def select_events_from_focal_region(df_binned, focal_cells,
                                    rank_col='event_min_storage_pct',
                                    ascending=True, n=None):
    """Select all events whose (sev_bin, mag_bin) falls in the focal region.

    Parameters
    ----------
    df_binned : pd.DataFrame
        Output of :func:`assign_grid_bins`.
    focal_cells : set of (i, j)
        From :func:`identify_focal_region`.
    rank_col : str
        Column to sort by.
    ascending : bool
    n : int or None
        If given, return only the top-*n* events.

    Returns
    -------
    pd.DataFrame sorted by *rank_col*.
    """
    import pandas as pd
    mask = pd.Series(False, index=df_binned.index)
    for i, j in focal_cells:
        mask |= (df_binned['sev_bin'] == i) & (df_binned['mag_bin'] == j)
    selected = df_binned[mask].sort_values(rank_col, ascending=ascending)
    if n is not None:
        selected = selected.head(n)
    return selected
