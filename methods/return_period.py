"""
Duration-adjusted drought return-period helpers.

Implements the Bonaccorso-Shiau interarrival-time formulation of drought
recurrence, and reports both the raw recurrence interval ``T_R`` and the
duration-adjusted "drought-free" interval ``T_W = T_R - E[D|bin]``:

  T_R  = E[L_interarrival] / P(class | drought)
       = (T_total / N_events) × (N_events / k_bin)
       = T_total / k_bin

  T_W  = T_R − E[duration | bin]

References
----------
- Loaiciga & Mariño (1991), J. Water Resour. Plan. Manage.
  — geophysical events as a renewal process; T_recurrence = E[D] + E[W].
- Fernández & Salas (1999), J. Hydrol. Eng.
  — return period as expected interarrival time.
- Shiau & Shen (2001), J. Water Resour. Plan. Manage.
  — drought recurrence using the run method.
- Bonaccorso, Cancelliere & Rossi (2003), Stoch. Environ. Res. Risk Assess.
  — analytical formulation; defines interarrival = drought + non-drought.
- Salas & Obeysekera (2014), J. Hydrol. Eng.  — review.

The drought-free interval ``T_W`` is what water-management planning
cares about: the expected number of *non-drought* years between the end
of one drought of the cell's severity-magnitude class and the start of
the next. For multi-year drought events this can be substantially less
than the nominal ``T_R``.
"""

import numpy as np

from methods.config import MIN_COUNT_PER_BIN

DAYS_PER_YEAR = 365.25


def compute_return_period_grid(df, sev_edges, mag_edges, n_years,
                                min_count=MIN_COUNT_PER_BIN):
    """Return per-bin recurrence interval, mean duration, and drought-free interval.

    Parameters
    ----------
    df : pd.DataFrame
        Event metrics. Must have columns ``severity``, ``magnitude``,
        ``realization_id``, ``duration_days``.
    sev_edges, mag_edges : np.ndarray
        Severity / magnitude bin edges.
    n_years : int
        Simulation years per realization.
    min_count : int
        Bins with fewer events are NaN.

    Returns
    -------
    T_R_grid : np.ndarray (ns, nm)
        Mean interarrival time per bin in years
        (= total_ensemble_years / count_bin).
    mean_duration_grid : np.ndarray (ns, nm)
        Mean drought duration per bin in years.
    T_W_grid : np.ndarray (ns, nm)
        Drought-free interval per bin in years (= T_R − E[D|bin]).
        NaN where T_W would be ≤ 0 (only possible with k_bin = 1 and a
        very long single event).
    count_grid : np.ndarray (ns, nm)
        Event count per bin (always populated, regardless of min_count).
    """
    sev = df['severity'].values
    mag = df['magnitude'].values
    dur_yr = df['duration_days'].values / DAYS_PER_YEAR

    n_realizations = df['realization_id'].nunique()
    total_years = n_realizations * n_years

    sev_idx = np.digitize(sev, sev_edges) - 1
    mag_idx = np.digitize(mag, mag_edges) - 1

    ns = len(sev_edges) - 1
    nm = len(mag_edges) - 1

    T_R_grid          = np.full((ns, nm), np.nan)
    mean_duration_grid = np.full((ns, nm), np.nan)
    T_W_grid          = np.full((ns, nm), np.nan)
    count_grid        = np.zeros((ns, nm), dtype=int)

    for i in range(ns):
        for j in range(nm):
            mask = (sev_idx == i) & (mag_idx == j)
            cnt = int(mask.sum())
            count_grid[i, j] = cnt
            if cnt < min_count:
                continue
            T_R = total_years / cnt
            E_D = float(dur_yr[mask].mean())
            T_W = T_R - E_D
            T_R_grid[i, j] = T_R
            mean_duration_grid[i, j] = E_D
            if T_W > 0:
                T_W_grid[i, j] = T_W

    return T_R_grid, mean_duration_grid, T_W_grid, count_grid
