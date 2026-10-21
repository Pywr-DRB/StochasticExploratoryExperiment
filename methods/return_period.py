"""
Duration-adjusted drought return-period helpers.

Three implementations are provided; the joint-exceedance version is the one
used for the figures and follows the bivariate return-period formulation in
the cited literature.

``compute_return_period_grid_exceedance``: joint-exceedance return period
    via the empirical copula. For each cell (i, j), evaluates the bivariate
    "AND" survival probability at the lower-left corner:

        P_exc(i,j) = P(S >= s_i AND M >= m_j)
                   = 1 - F_S(s_i) - F_M(m_j) + C_n(F_S(s_i), F_M(m_j))

    This is the formulation used by Shiau and Shen (2001), Bonaccorso et
    al. (2003), Shiau (2006), and Salvadori and De Michele (2004, 2010);
    the empirical copula C_n (Deheuvels 1979; Nelsen 2006, Sect. 2.4)
    provides the nonparametric estimate of the joint distribution.

``compute_return_period_grid_copula``: per-bin probability via the
    empirical-copula inclusion-exclusion; equivalent to
    ``compute_return_period_grid`` for fixed grid edges. Retained for
    comparison.

``compute_return_period_grid``: per-bin rate inversion,
    P(class) = k_bin / n_total. Retained for comparison.

All three apply the Bonaccorso-Shiau interarrival-time formula:

  T_R  = E[L_interarrival] / P(class | drought)
       = (T_total / N_events) / P

and the duration-adjusted drought-free interval:

  T_W  = T_R - E[duration | conditioning region]

For the joint-exceedance variant, ``E[duration | exc]`` is the mean
duration over the exceedance region. For the per-bin variants, it is
the mean duration over events in the bin.

References
----------
- Loaiciga and Mariño (1991), J. Water Resour. Plan. Manage.:
  geophysical events as a renewal process; T_recurrence = E[D] + E[W].
- Fernández and Salas (1999), J. Hydrol. Eng.:
  return period as expected interarrival time.
- Shiau and Shen (2001), J. Water Resour. Plan. Manage.:
  drought recurrence using the run method.
- Bonaccorso, Cancelliere and Rossi (2003), Stoch. Environ. Res. Risk Assess.:
  analytical formulation; defines interarrival = drought + non-drought.
- Salas and Obeysekera (2014), J. Hydrol. Eng.: review.
- Deheuvels (1979), Stat. Ind. Acad. Publ. Bull.:
  empirical copula definition.
- Nelsen (2006), An Introduction to Copulas, Sect. 2.4.
- Genest and Favre (2007), J. Hydrol. Eng.:
  empirical copula for bivariate hydrological frequency analysis.
- Salvadori and De Michele (2004), Water Resour. Res.:
  bivariate copula return periods (AND/OR formulations).
- Salvadori and De Michele (2010), J. Hydrol.:
  multivariate real-time assessment of droughts via copulas.
- Shiau (2006), Water Resour. Manage.:
  bivariate return period for drought severity and duration.

``T_W`` is the expected number of non-drought years between the end of one
drought of the cell's severity-magnitude class and the start of the next;
for multi-year events it can be substantially less than ``T_R``.
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
        Drought-free interval per bin in years (= T_R - E[D|bin]).
        NaN where T_W would be <= 0 (only possible with k_bin = 1 and a
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


# ---------------------------------------------------------------------------
# Joint-exceedance (bivariate AND) return period
# ---------------------------------------------------------------------------

def compute_return_period_grid_exceedance(df, sev_edges, mag_edges, n_years,
                                           min_count=MIN_COUNT_PER_BIN):
    """Joint-exceedance return-period grids via the empirical copula.

    For each grid cell (i, j), evaluates the bivariate "AND" exceedance
    probability at the *lower-left corner* of the cell, using the
    survival-copula decomposition:

        P_exc(i,j) = P(S >= sev_edges[i]  AND  M >= mag_edges[j])
                   = 1 - F_S(s_i) - F_M(m_j) + C_n(F_S(s_i), F_M(m_j))

    where F_S, F_M are empirical marginal CDFs and C_n is the empirical
    copula (Deheuvels 1979; Nelsen 2006, Sect. 2.4). The Bonaccorso-Shiau
    formula then gives the recurrence interval and its duration-adjusted
    counterpart (Shiau and Shen 2001; Bonaccorso et al. 2003; Shiau 2006;
    Salvadori and De Michele 2004, 2010):

        T_R(i,j) = E[L_inter] / P_exc(i,j)
        T_W(i,j) = T_R(i,j) - E[D | sev >= s_i AND mag >= m_j]

    The cell color on the heatmap represents "the drought-free interval
    between events at least as severe and as long as the lower-left
    corner of this cell." The gradient is monotone: lower-left cells
    (frequent moderate droughts) give short T_W; upper-right cells
    (rare extreme droughts) give long T_W.

    Parameters
    ----------
    df : pd.DataFrame
        Event metrics with columns ``severity``, ``magnitude``,
        ``realization_id``, ``duration_days``.
    sev_edges, mag_edges : np.ndarray
        Bin edges from
        :func:`methods.plotting.heatmap.make_shared_edges_logmag`.
    n_years : int
        Simulation years per realization.
    min_count : int
        Minimum exceedance count for a cell to be reported (else NaN).

    Returns
    -------
    T_R_grid, mean_duration_grid, T_W_grid : np.ndarray (ns, nm)
        Joint-exceedance metrics evaluated at the lower-left corner of
        each cell.
    count_grid : np.ndarray (ns, nm)
        Per-bin event count (used for hatching). The min_count threshold
        is applied to the exceedance count at each corner, not to this
        per-bin count.
    """
    from scipy.stats import rankdata

    sev = df['severity'].values
    mag = df['magnitude'].values
    dur_yr = df['duration_days'].values / DAYS_PER_YEAR

    n = len(sev)
    n_real = df['realization_id'].nunique()
    total_years = n_real * n_years
    E_L_inter = total_years / n   # mean interarrival time (years)

    # Pseudo-observations on (0, 1); Weibull plotting positions keep the
    # rank transform off the {0, 1} boundary (Genest and Favre 2007, Sect. 2).
    u = rankdata(sev) / (n + 1)
    v = rankdata(mag) / (n + 1)

    # Empirical marginal CDFs at the bin edges.
    sev_sorted = np.sort(sev)
    mag_sorted = np.sort(mag)
    F_S_edges = np.searchsorted(sev_sorted, sev_edges, side='right') / n
    F_M_edges = np.searchsorted(mag_sorted, mag_edges, side='right') / n

    # Empirical copula evaluated at every bin-edge corner.
    # Salvadori & De Michele (2010); Genest & Favre (2007).
    C = _empirical_copula_at_corners(u, v, F_S_edges, F_M_edges)

    # Joint-exceedance probability at the lower-left corner of cell (i,j):
    # survival copula via inclusion-exclusion (Salvadori & De Michele 2004).
    F_S_grid = F_S_edges[:-1][:, None]   # shape (ns, 1)
    F_M_grid = F_M_edges[:-1][None, :]   # shape (1, nm)
    P_exc = 1.0 - F_S_grid - F_M_grid + C[:-1, :-1]

    ns = len(sev_edges) - 1
    nm = len(mag_edges) - 1

    sev_idx = np.digitize(sev, sev_edges) - 1
    mag_idx = np.digitize(mag, mag_edges) - 1

    T_R_grid           = np.full((ns, nm), np.nan)
    mean_duration_grid = np.full((ns, nm), np.nan)
    T_W_grid           = np.full((ns, nm), np.nan)
    count_grid         = np.zeros((ns, nm), dtype=int)

    for i in range(ns):
        for j in range(nm):
            # Per-bin count (used for hatching)
            mask_bin = (sev_idx == i) & (mag_idx == j)
            count_grid[i, j] = int(mask_bin.sum())

            # Exceedance region: sev >= sev_edges[i] AND mag >= mag_edges[j].
            mask_exc = (sev >= sev_edges[i]) & (mag >= mag_edges[j])
            n_exc = int(mask_exc.sum())
            if n_exc < min_count:
                continue

            p = float(P_exc[i, j])
            if p <= 0:
                continue

            T_R = E_L_inter / p
            E_D = float(dur_yr[mask_exc].mean())
            T_W = T_R - E_D
            T_R_grid[i, j] = T_R
            mean_duration_grid[i, j] = E_D
            if T_W > 0:
                T_W_grid[i, j] = T_W

    return T_R_grid, mean_duration_grid, T_W_grid, count_grid


# ---------------------------------------------------------------------------
# Empirical-copula (checkerboard) implementation
# ---------------------------------------------------------------------------

def _empirical_copula_at_corners(u, v, u_edges, v_edges):
    """Evaluate the empirical copula C_n at all grid corner points.

    C_n(ue, ve) is the fraction of events with rank_S <= ue and rank_M <= ve
    (Deheuvels 1979; Nelsen 2006, Sect. 2.4), evaluated at every
    (ns+1) x (nm+1) corner; the inner loop is vectorised over the n events,
    so the cost is O((ns+1)*(nm+1)*n).

    Parameters
    ----------
    u, v : np.ndarray, shape (n,)
        Rank-transformed marginals in (0, 1)  (Weibull plotting positions).
    u_edges, v_edges : np.ndarray
        Marginal-CDF values at the (ns+1) severity and (nm+1) magnitude
        bin edges respectively.

    Returns
    -------
    C : np.ndarray, shape (ns+1, nm+1)
        C[i, j] = C_n(u_edges[i], v_edges[j]).
    """
    ns1 = len(u_edges)
    nm1 = len(v_edges)
    C = np.zeros((ns1, nm1))
    for i in range(ns1):
        mask_u = u <= u_edges[i]
        for j in range(nm1):
            C[i, j] = np.mean(mask_u & (v <= v_edges[j]))
    return C


def compute_return_period_grid_copula(df, sev_edges, mag_edges, n_years,
                                       min_count=MIN_COUNT_PER_BIN):
    """Empirical-copula (checkerboard) return-period grids.

    Replaces the raw per-bin count ``P = k_bin / n_total`` used by
    :func:`compute_return_period_grid` with the checkerboard empirical
    copula probability mass for each bin (Deheuvels 1979; Nelsen 2006, Sect. 2.4).
    This accounts for the positive dependence between severity and magnitude
    (magnitude is approximately severity x duration, so the two co-move strongly in the right tail)
    and uses marginal information from all events, improving stability for
    sparse high-severity bins.

    T_R and T_W follow the Bonaccorso-Shiau interarrival-time formula
    (Bonaccorso et al. 2003; Shiau and Shen 2001):

        T_R(i,j) = E[L_inter] / P_bin_copula(i,j)
        T_W(i,j) = T_R(i,j) - E[D | bin]

    Signature and return values are identical to
    :func:`compute_return_period_grid`.

    Parameters
    ----------
    df : pd.DataFrame
        Event metrics with columns ``severity``, ``magnitude``,
        ``realization_id``, ``duration_days``.
    sev_edges, mag_edges : np.ndarray
        Bin edges from :func:`methods.plotting.heatmap.make_shared_edges_logmag`.
    n_years : int
        Simulation years per realization.
    min_count : int
        Bins with fewer events are NaN.

    Returns
    -------
    T_R_grid : np.ndarray (ns, nm)
    mean_duration_grid : np.ndarray (ns, nm)
    T_W_grid : np.ndarray (ns, nm)
    count_grid : np.ndarray (ns, nm)
    """
    from scipy.stats import rankdata

    sev = df['severity'].values
    mag = df['magnitude'].values
    dur_yr = df['duration_days'].values / DAYS_PER_YEAR

    n = len(sev)
    n_real = df['realization_id'].nunique()
    total_years = n_real * n_years
    E_L_inter = total_years / n   # mean interarrival time (years)

    # Rank-transform to copula space; the Weibull plotting position avoids the
    # 0/1 boundary and keeps C_n from saturating at the upper-right corner
    # (Genest and Favre 2007, Sect. 2).
    u = rankdata(sev) / (n + 1)
    v = rankdata(mag) / (n + 1)

    # Empirical marginal CDFs at each bin edge: F_S(s) = #{sev <= s} / n
    sev_sorted = np.sort(sev)
    mag_sorted = np.sort(mag)
    u_edges = np.searchsorted(sev_sorted, sev_edges, side='right') / n
    v_edges = np.searchsorted(mag_sorted, mag_edges, side='right') / n

    # Empirical copula at all (ns+1) x (nm+1) corner points, then
    # inclusion-exclusion to get the bin probability mass:
    # P_bin[i,j] = C_n(i+1,j+1) - C_n(i+1,j) - C_n(i,j+1) + C_n(i,j)
    # (Salvadori and De Michele 2010; Genest and Favre 2007).
    C = _empirical_copula_at_corners(u, v, u_edges, v_edges)
    P_bin = C[1:, 1:] - C[1:, :-1] - C[:-1, 1:] + C[:-1, :-1]

    ns = len(sev_edges) - 1
    nm = len(mag_edges) - 1

    sev_idx = np.digitize(sev, sev_edges) - 1
    mag_idx = np.digitize(mag, mag_edges) - 1

    T_R_grid           = np.full((ns, nm), np.nan)
    mean_duration_grid = np.full((ns, nm), np.nan)
    T_W_grid           = np.full((ns, nm), np.nan)
    count_grid         = np.zeros((ns, nm), dtype=int)

    for i in range(ns):
        for j in range(nm):
            mask = (sev_idx == i) & (mag_idx == j)
            cnt = int(mask.sum())
            count_grid[i, j] = cnt
            if cnt < min_count:
                continue
            p = float(P_bin[i, j])
            if p <= 0:
                continue
            T_R = E_L_inter / p
            E_D = float(dur_yr[mask].mean())
            T_W = T_R - E_D
            T_R_grid[i, j] = T_R
            mean_duration_grid[i, j] = E_D
            if T_W > 0:
                T_W_grid[i, j] = T_W

    return T_R_grid, mean_duration_grid, T_W_grid, count_grid
