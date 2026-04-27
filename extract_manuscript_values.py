"""
Extract manuscript reference values for Section 4 from the full 2000-member ensemble.

Writes (under manuscript/ensemble_stats/):
  extracted_values_rev1.json         -- all numeric keys
  extracted_values_rev1.md           -- alphabetical human-readable listing
  extracted_values_by_section.md     -- section-grouped listing
  focal_event_selections.json        -- Section 4.6 focal event draws
  extracted_values_rev1_errors.log   -- any failures

Run from the experiment repo root (or via sbatch S8_extract_manuscript_values.sh):
  python extract_manuscript_values.py
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from methods.config import (
    BASELINE_DATASET,
    DATASET_CONFIGS,
    MGD_TO_MCM,
    N_YEARS,
    NYC_RESERVOIRS,
    NYC_TOTAL_CAPACITY,
    RECONSTRUCTION_N_YEARS,
    SSI_WINDOWS,
    SSI_NODE,
    OUTPUT_DIR,
)
from methods.load import (
    compute_event_exceedances,
    load_and_process_historical_models,
    load_baseline_historical_flow,
    load_contribution_metrics,
    load_drought_events,
    load_event_metrics,
    load_ffmp_boundaries,
    load_zone_probabilities,
)
from methods.water_year import vectorized_water_year, vectorized_water_year_doy
from methods.zone_duration_metrics import calculate_drought_zone_events

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
MS = ROOT / "manuscript"
MS_OUT = MS / "ensemble_stats"
MS_OUT.mkdir(parents=True, exist_ok=True)

LOG_FILE = MS_OUT / "extracted_values_rev1_errors.log"

# Dataset id mappings  (prefix -> dataset_id)
ENS_IDS: dict[str, str] = {
    "ssb":  "stationary_ensemble",
    "wwds": "climate_adjusted_low",
    "wwss": "climate_adjusted_high",
}

# Set up error log
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
_error_log: list[str] = []   # also kept in memory for summary


def _log_error(key: str, exc: Exception) -> None:
    msg = f"FAILED {key}: {type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    logging.warning(msg)
    _error_log.append(f"  {key}: {type(exc).__name__}: {exc}")


def _safe(out: dict, key: str, func):
    """
    Call func(), store result in out[key].
    On any exception, store np.nan and log the failure.
    """
    try:
        result = func()
        out[key] = result
    except Exception as exc:
        out[key] = np.nan
        _log_error(key, exc)


# ---------------------------------------------------------------------------
# ACF helper
# ---------------------------------------------------------------------------

def _compute_acf_per_year(series: pd.Series, max_lag: int = 30) -> pd.DataFrame:
    """Compute per-water-year ACF up to max_lag.

    Returns a DataFrame with rows = lag (0..max_lag) and columns = water years.
    """
    try:
        from statsmodels.tsa.stattools import acf as sm_acf
    except ImportError:
        # fall back to numpy autocorrelation
        sm_acf = None

    results = {}
    water_years = vectorized_water_year(series.index)
    unique_wys = np.unique(water_years)
    for wy in unique_wys:
        group = series[water_years == wy].dropna()
        if len(group) < max_lag + 5:
            continue
        if sm_acf is not None:
            vals = sm_acf(group.values, nlags=max_lag, fft=True)
        else:
            # manual ACF via numpy correlate
            x = group.values - group.values.mean()
            full = np.correlate(x, x, mode="full")
            full = full[len(x) - 1:]
            vals = full[:max_lag + 1] / full[0]
        results[wy] = vals
    return pd.DataFrame(results)  # rows = lags 0..max_lag, cols = years


# ---------------------------------------------------------------------------
# Section 4.1 — Ensemble validation
# ---------------------------------------------------------------------------

def section_41(out: dict) -> dict:
    print("Section 4.1: ensemble validation...")

    # --- Reconstruction flow ---
    try:
        rec_hist = load_baseline_historical_flow(
            gage_flow=False, period="full", flowtype=BASELINE_DATASET
        )
        nyc_flow = rec_hist[NYC_RESERVOIRS].sum(axis=1)

        acf_df = _compute_acf_per_year(nyc_flow, max_lag=30)
        _safe(out, "hist_acf_lag1",  lambda: float(acf_df.iloc[1].median()))
        _safe(out, "hist_acf_lag30", lambda: float(acf_df.iloc[30].median()))
    except Exception as exc:
        _log_error("hist_acf_lag1/lag30", exc)
        out.setdefault("hist_acf_lag1", np.nan)
        out.setdefault("hist_acf_lag30", np.nan)

    # --- max_abs_median_acf_offset: requires loading ensemble inflow, heavy ---
    # We compute reconstruction median ACF at each lag, and compare to it using
    # the per-realization inflow loaded from the postprocessed HDF5.  Because
    # loading 2000 x 70 yr daily inflows is very expensive this is wrapped in
    # try/except; it will fail gracefully if the HDF5 is unavailable.
    try:
        import pywrdrb
        from methods.config import RECONSTRUCTION_OUTPUT_FNAME

        # Reconstruct median ACF (lags 1..30) from above
        rec_hist_full = load_baseline_historical_flow(
            gage_flow=False, period="full", flowtype=BASELINE_DATASET
        )
        nyc_flow_full = rec_hist_full[NYC_RESERVOIRS].sum(axis=1)
        rec_acf_df = _compute_acf_per_year(nyc_flow_full, max_lag=30)
        rec_median_acf = rec_acf_df.median(axis=1).values[1:31]  # lags 1-30

        # Load ensemble inflow
        fname = f"{OUTPUT_DIR}/stationary_ensemble_with_postprocessing.hdf5"
        data_ens = pywrdrb.Data()
        data_ens.load_from_export(fname, results_sets=["inflow"])
        realizations = sorted(data_ens.inflow["stationary_ensemble"].keys())

        ens_acf_per_lag: list[list[float]] = [[] for _ in range(30)]
        for r in realizations:
            inflow_r = data_ens.inflow["stationary_ensemble"][r][NYC_RESERVOIRS].sum(axis=1)
            r_acf_df = _compute_acf_per_year(inflow_r, max_lag=30)
            lag_medians = r_acf_df.median(axis=1).values[1:31]
            for lag_idx in range(30):
                ens_acf_per_lag[lag_idx].append(lag_medians[lag_idx])

        ens_median_acf = np.array([np.median(v) for v in ens_acf_per_lag])
        acf_offsets = np.abs(ens_median_acf - rec_median_acf)
        out["max_abs_median_acf_offset"] = float(acf_offsets.max())
        # 1-indexed lag (in days) at which the maximum offset occurs
        out["max_abs_median_acf_offset_lag"] = int(np.argmax(acf_offsets) + 1)
    except Exception as exc:
        _log_error("max_abs_median_acf_offset", exc)
        out["max_abs_median_acf_offset"] = np.nan
        out["max_abs_median_acf_offset_lag"] = np.nan

    # --- FDC values from annual FDCs ---
    try:
        rec_hist_fdc = load_baseline_historical_flow(
            gage_flow=False, period="full", flowtype=BASELINE_DATASET
        )
        nyc_fdc_flow = rec_hist_fdc[NYC_RESERVOIRS].sum(axis=1)
        water_years_fdc = vectorized_water_year(nyc_fdc_flow.index)
        unique_wys_fdc = np.unique(water_years_fdc)

        fdc_high_vals: list[float] = []
        fdc_low_vals: list[float] = []
        for wy in unique_wys_fdc:
            wy_vals = np.sort(nyc_fdc_flow[water_years_fdc == wy].dropna().values)[::-1]
            if len(wy_vals) < 10:
                continue
            exceedance = np.arange(1, len(wy_vals) + 1) / (len(wy_vals) + 1)
            fdc_high_vals.append(float(np.interp(0.01, exceedance, wy_vals)))
            fdc_low_vals.append(float(np.interp(0.99, exceedance, wy_vals)))

        # FDC values are aggregated in MGD; convert to MCM/d for the manuscript.
        out["hist_fdc_high"] = float(np.median(fdc_high_vals)) * MGD_TO_MCM
        out["hist_fdc_low"]  = float(np.median(fdc_low_vals))  * MGD_TO_MCM
    except Exception as exc:
        _log_error("hist_fdc_high/low", exc)
        out.setdefault("hist_fdc_high", np.nan)
        out.setdefault("hist_fdc_low", np.nan)

    # --- Summer low-flow tail gap (weeks 30-40 of water year) ---
    try:
        rec_weekly_min: dict[int, float] = {}
        nyc_doy_fdc = vectorized_water_year_doy(nyc_fdc_flow.index)
        for wk in range(30, 41):
            day_start = (wk - 1) * 7 + 1
            day_end = wk * 7
            mask = (nyc_doy_fdc >= day_start) & (nyc_doy_fdc <= day_end)
            vals = nyc_fdc_flow[mask].dropna()
            rec_weekly_min[wk] = float(vals.min()) if len(vals) > 0 else np.nan

        # For the ensemble we use the already-computed zone_probabilities as a proxy;
        # a true ensemble weekly distribution requires loading the HDF5.
        # We attempt the full computation but fall back to np.nan if HDF5 unavailable.
        import pywrdrb
        fname_ens = f"{OUTPUT_DIR}/stationary_ensemble_with_postprocessing.hdf5"
        data_ens2 = pywrdrb.Data()
        data_ens2.load_from_export(fname_ens, results_sets=["inflow"])
        realizations2 = sorted(data_ens2.inflow["stationary_ensemble"].keys())

        ens_wk_vals: dict[int, list[float]] = {wk: [] for wk in range(30, 41)}
        for r in realizations2:
            inflow_r2 = data_ens2.inflow["stationary_ensemble"][r][NYC_RESERVOIRS].sum(axis=1)
            doy_r = vectorized_water_year_doy(inflow_r2.index)
            for wk in range(30, 41):
                day_start = (wk - 1) * 7 + 1
                day_end = wk * 7
                mask_r = (doy_r >= day_start) & (doy_r <= day_end)
                ens_wk_vals[wk].extend(inflow_r2[mask_r].dropna().tolist())

        gaps: list[float] = []
        for wk in range(30, 41):
            rec_min = rec_weekly_min.get(wk, np.nan)
            if np.isnan(rec_min) or rec_min <= 0 or len(ens_wk_vals[wk]) == 0:
                continue
            ens_q005 = float(np.nanpercentile(ens_wk_vals[wk], 0.5))
            gap_pct = 100.0 * (rec_min - ens_q005) / rec_min
            gaps.append(gap_pct)

        out["summer_lf_tail_gap_pct"] = float(max(gaps)) if gaps else np.nan
    except Exception as exc:
        _log_error("summer_lf_tail_gap_pct", exc)
        out.setdefault("summer_lf_tail_gap_pct", np.nan)

    return out


# ---------------------------------------------------------------------------
# Section 4.2 — Drought emergence
# ---------------------------------------------------------------------------

def _compute_ssi_zone_coverage(dataset_id: str, ssi_window: int) -> float:
    """
    Fraction of realization-months in drought zone (Watch/Warning/Emergency)
    AND an SSI-W event is active, for the given dataset.
    """
    events = load_drought_events(dataset_id, ssi_window)
    zp = load_zone_probabilities(dataset_id, period="weekly")
    if zp is None:
        return np.nan

    # Build a boolean mask over all ensemble realization-months using events.
    # We approximate using the event-level fraction of weeks in drought zone
    # by counting per-realization active weeks that fall in drought zone.

    # Get all unique realizations in events
    realizations = events["realization_id"].unique()
    n_reals = len(realizations)

    # For each realization, compute fraction of weeks that are both:
    #   (a) inside a SSI event window
    #   (b) in a drought zone
    # We use zone probabilities as a population-level estimate rather than
    # reloading full storage timeseries.  Specifically:
    #   coverage ≈ (fraction of realization-months active in SSI event) *
    #              (fraction of those months where zone is stressed)
    # Simpler approach: for each realization count event months / total months,
    # then weight by the zone_probs stressed fraction over those weeks.
    # The most direct implementable approach without loading raw storage:
    # count total event-days across ensemble / total ensemble-days, restricted
    # to stressed zone using zone probabilities as a proxy weight.

    total_months = n_reals * N_YEARS * 12  # approximate
    event_months = 0
    for _, row in events.iterrows():
        start = pd.Timestamp(row["start"])
        end = pd.Timestamp(row["end"])
        duration_months = max(1, round((end - start).days / 30.44))
        event_months += duration_months

    # Stressed zone probability from zone_probs (average over all weeks)
    if "any_drought" in zp.columns:
        stressed_prob = float(zp["any_drought"].mean()) / 100.0
    else:
        cols = [c for c in zp.columns if any(
            z in c.lower() for z in ["watch", "warning", "emergency",
                                      "zone_4", "zone_5", "zone_6", "4", "5", "6"]
        )]
        if cols:
            stressed_prob = float(zp[cols].sum(axis=1).mean()) / 100.0
        else:
            # Fall back: sum zone columns > 3 if they are named by zone number
            numeric_cols = zp.select_dtypes(include=[np.number]).columns
            zone_cols = [c for c in numeric_cols if str(c) in ["4", "5", "6"]]
            stressed_prob = float(zp[zone_cols].sum(axis=1).mean()) / 100.0 if zone_cols else np.nan

    coverage_pct = 100.0 * (event_months / total_months) * stressed_prob
    return float(coverage_pct)


def section_42(out: dict) -> dict:
    print("Section 4.2: SSI drought emergence...")

    # --- Reconstruction 1960s event ---
    try:
        rec_events = load_drought_events("reconstruction", 3, observed=True)
        # 1960s event: started in 1963 or 1964 — find by max severity in 1960s
        mask_1960s = (rec_events["start"].dt.year >= 1963) & (rec_events["start"].dt.year <= 1965)
        ev1960 = rec_events[mask_1960s].sort_values("severity", ascending=False).iloc[0]
        out["hist_1960s_ssi3_severity"]  = float(ev1960["severity"])
        out["hist_1960s_ssi3_magnitude"] = float(ev1960["magnitude"])
        out["hist_1960s_ssi3_duration"]  = int(ev1960["duration"]) if "duration" in ev1960.index else int(
            (pd.Timestamp(ev1960["end"]) - pd.Timestamp(ev1960["start"])).days
        )
        out["hist_1960s_event_id"] = int(ev1960.name)
    except Exception as exc:
        _log_error("hist_1960s_ssi3_*", exc)
        out.setdefault("hist_1960s_ssi3_severity", np.nan)
        out.setdefault("hist_1960s_ssi3_magnitude", np.nan)
        out.setdefault("hist_1960s_ssi3_duration", np.nan)
        out.setdefault("hist_1960s_event_id", np.nan)

    # --- SSI timescale zone coverage ---
    for W in SSI_WINDOWS:
        key = f"ssi{W}_zone_coverage_pct"
        _safe(out, key, lambda W=W: _compute_ssi_zone_coverage("stationary_ensemble", W))

    # --- Per-ensemble tail counts and exceedance rates ---
    # Cache all events to avoid reloading
    all_events: dict[str, pd.DataFrame] = {}
    for key, ens_id in ENS_IDS.items():
        try:
            all_events[key] = load_drought_events(ens_id, 3)
        except Exception as exc:
            _log_error(f"load_drought_events {key}", exc)
            all_events[key] = pd.DataFrame()

    sev_ref = out.get("hist_1960s_ssi3_severity", np.nan)
    mag_ref = out.get("hist_1960s_ssi3_magnitude", np.nan)

    for key, events in all_events.items():
        if events.empty:
            continue

        # Tail counts exceeding 1960s
        count_key_sev = ("count_events_more_severe_than_1960s"
                         if key == "ssb" else f"{key}_count_events_more_severe_than_1960s")
        count_key_mag = ("count_events_more_magnitude_than_1960s"
                         if key == "ssb" else f"{key}_count_events_more_magnitude_than_1960s")
        _safe(out, count_key_sev,
              lambda e=events, t=sev_ref: int((e["severity"] > t).sum()))
        _safe(out, count_key_mag,
              lambda e=events, t=mag_ref: int((e["magnitude"] > t).sum()))

        # Canonical severity exceedance rates
        all_real_ids = events["realization_id"].unique()
        n_total_real = max(len(all_real_ids), 1)
        realization_index = pd.RangeIndex(n_total_real)

        for sev in [1.0, 1.5, 2.0, 2.5, 3.0]:
            s_str = f"{sev:.1f}".replace(".", "")

            def _sev_rates(e=events, s=sev, ri=realization_index):
                per_real = (e.groupby("realization_id")
                             .apply(lambda g: int((g["severity"] >= s).sum()))
                             .reindex(ri, fill_value=0))
                return per_real / N_YEARS

            try:
                rates = _sev_rates()
                out[f"{key}_median_exrate_sev{s_str}"] = float(rates.median())
                out[f"{key}_q25_exrate_sev{s_str}"]    = float(rates.quantile(0.25))
                out[f"{key}_q75_exrate_sev{s_str}"]    = float(rates.quantile(0.75))
                out[f"{key}_q005_exrate_sev{s_str}"]   = float(rates.quantile(0.005))
                out[f"{key}_q995_exrate_sev{s_str}"]   = float(rates.quantile(0.995))
                med = float(rates.median())
                out[f"{key}_rp_sev{s_str}"] = float(1.0 / med) if med > 0 else np.inf
                out[f"{key}_min_count_sev{s_str}"] = int((rates * N_YEARS).min())
                out[f"{key}_max_count_sev{s_str}"] = int((rates * N_YEARS).max())
            except Exception as exc:
                for out_key in [
                    f"{key}_median_exrate_sev{s_str}",
                    f"{key}_q25_exrate_sev{s_str}",
                    f"{key}_q75_exrate_sev{s_str}",
                    f"{key}_q005_exrate_sev{s_str}",
                    f"{key}_q995_exrate_sev{s_str}",
                    f"{key}_rp_sev{s_str}",
                    f"{key}_min_count_sev{s_str}",
                    f"{key}_max_count_sev{s_str}",
                ]:
                    _log_error(out_key, exc)
                    out[out_key] = np.nan

        # Canonical magnitude exceedance rates
        for mag in [5, 10, 20, 40]:
            def _mag_rates(e=events, m=mag, ri=realization_index):
                per_real = (e.groupby("realization_id")
                             .apply(lambda g: int((g["magnitude"] >= m).sum()))
                             .reindex(ri, fill_value=0))
                return per_real / N_YEARS

            try:
                rates_m = _mag_rates()
                out[f"{key}_median_exrate_mag{mag}"] = float(rates_m.median())
                med_m = float(rates_m.median())
                out[f"{key}_rp_mag{mag}"] = float(1.0 / med_m) if med_m > 0 else np.inf
            except Exception as exc:
                _log_error(f"{key}_median_exrate_mag{mag}", exc)
                out[f"{key}_median_exrate_mag{mag}"] = np.nan
                out[f"{key}_rp_mag{mag}"] = np.nan

    # --- Text aliases: sev2.0 -> sev20 key already created; add sev2 shorthand ---
    for k_pattern, k_alias in [
        ("median_exrate_sev20", "median_exrate_sev2"),
        ("q005_exrate_sev20",   "q005_exrate_sev2"),
        ("q995_exrate_sev20",   "q995_exrate_sev2"),
        ("min_count_sev20",     "min_count_sev2"),
        ("max_count_sev20",     "max_count_sev2"),
        ("rp_sev20",            "rp_sev2"),
    ]:
        src = f"ssb_{k_pattern}"
        dst = f"ssb_{k_alias}"
        if src in out and dst not in out:
            out[dst] = out[src]

    # --- SSB 1960s exceedance rate ---
    try:
        ssb_ev = all_events.get("ssb", pd.DataFrame())
        if not ssb_ev.empty and not np.isnan(sev_ref):
            ri_ssb = pd.RangeIndex(ssb_ev["realization_id"].nunique())
            per_real_1960 = (ssb_ev.groupby("realization_id")
                              .apply(lambda g: int((g["severity"] >= sev_ref).sum()))
                              .reindex(ri_ssb, fill_value=0))
            rates_1960 = per_real_1960 / N_YEARS
            med_1960 = float(rates_1960.median())
            out["ssb_ex_rate_1960s"] = med_1960
            out["ssb_rp_1960s"] = float(1.0 / med_1960) if med_1960 > 0 else np.inf
        else:
            out["ssb_ex_rate_1960s"] = np.nan
            out["ssb_rp_1960s"] = np.nan
    except Exception as exc:
        _log_error("ssb_ex_rate_1960s", exc)
        out["ssb_ex_rate_1960s"] = np.nan
        out["ssb_rp_1960s"] = np.nan

    # --- WWSS-specific tail keys ---
    try:
        wwss_ev = all_events.get("wwss", pd.DataFrame())
        out["wwss_severity_tail_cutoff"]  = float(wwss_ev["severity"].max()) if not wwss_ev.empty else np.nan
        out["wwss_magnitude_tail_cutoff"] = float(wwss_ev["magnitude"].max()) if not wwss_ev.empty else np.nan
        out["wwss_sev_thresh_minus_1"] = float(sev_ref - 1.0) if not np.isnan(sev_ref) else np.nan
        thresh_m1 = out["wwss_sev_thresh_minus_1"]
        out["wwss_count_events_more_severe_than_hist1960s_minus_1"] = (
            int((wwss_ev["severity"] >= thresh_m1).sum())
            if not wwss_ev.empty and not np.isnan(thresh_m1) else np.nan
        )
    except Exception as exc:
        _log_error("wwss_tail_keys", exc)
        for k in ["wwss_severity_tail_cutoff", "wwss_magnitude_tail_cutoff",
                  "wwss_sev_thresh_minus_1",
                  "wwss_count_events_more_severe_than_hist1960s_minus_1"]:
            out.setdefault(k, np.nan)

    # --- Climate-scenario shift in exceedance rates (mild severity band 1.0-1.5) ---
    try:
        ssb_ev = all_events.get("ssb", pd.DataFrame())
        wwds_ev = all_events.get("wwds", pd.DataFrame())
        wwss_ev = all_events.get("wwss", pd.DataFrame())

        def _mild_rate_per_real(ev: pd.DataFrame) -> pd.Series:
            ri = pd.RangeIndex(ev["realization_id"].nunique())
            per_real = (ev.groupby("realization_id")
                         .apply(lambda g: int(((g["severity"] >= 1.0) & (g["severity"] < 1.5)).sum()))
                         .reindex(ri, fill_value=0))
            return per_real / N_YEARS

        if not ssb_ev.empty and not wwds_ev.empty:
            r_ssb_mild = _mild_rate_per_real(ssb_ev)
            r_wwds_mild = _mild_rate_per_real(wwds_ev)
            out["wwds_delta_exrate_mild"] = float(r_wwds_mild.median() - r_ssb_mild.median())
        else:
            out["wwds_delta_exrate_mild"] = np.nan

        if not ssb_ev.empty and not wwss_ev.empty:
            r_wwss_mild = _mild_rate_per_real(wwss_ev)
            out["wwss_delta_exrate_mild"] = float(r_wwss_mild.median() - r_ssb_mild.median())
            out["wwss_q005_delta_exrate_mild"] = float(
                r_wwss_mild.quantile(0.005) - r_ssb_mild.quantile(0.005))
            out["wwss_q995_delta_exrate_mild"] = float(
                r_wwss_mild.quantile(0.995) - r_ssb_mild.quantile(0.995))
        else:
            out["wwss_delta_exrate_mild"] = np.nan
            out["wwss_q005_delta_exrate_mild"] = np.nan
            out["wwss_q995_delta_exrate_mild"] = np.nan

        # wwds_magnitude_threshold_zero: smallest magnitude where
        # (WWDS median - SSB median) exceedance rate crosses zero from positive to negative
        if not ssb_ev.empty and not wwds_ev.empty:
            mag_thresholds = np.arange(1, 101, 1)
            ri_ssb = pd.RangeIndex(ssb_ev["realization_id"].nunique())
            ri_wwds = pd.RangeIndex(wwds_ev["realization_id"].nunique())
            zero_cross = np.nan
            prev_delta = np.nan
            for m in mag_thresholds:
                ssb_r = (ssb_ev.groupby("realization_id")
                          .apply(lambda g, _m=m: int((g["magnitude"] >= _m).sum()))
                          .reindex(ri_ssb, fill_value=0)).mean() / N_YEARS
                wwds_r = (wwds_ev.groupby("realization_id")
                           .apply(lambda g, _m=m: int((g["magnitude"] >= _m).sum()))
                           .reindex(ri_wwds, fill_value=0)).mean() / N_YEARS
                delta = wwds_r - ssb_r
                if not np.isnan(prev_delta) and prev_delta > 0 >= delta:
                    zero_cross = float(m)
                    break
                prev_delta = delta
            out["wwds_magnitude_threshold_zero"] = zero_cross
        else:
            out["wwds_magnitude_threshold_zero"] = np.nan

    except Exception as exc:
        _log_error("climate_scenario_shift_keys", exc)
        for k in ["wwds_delta_exrate_mild", "wwds_magnitude_threshold_zero",
                  "wwss_delta_exrate_mild", "wwss_q005_delta_exrate_mild",
                  "wwss_q995_delta_exrate_mild"]:
            out.setdefault(k, np.nan)

    return out


# ---------------------------------------------------------------------------
# Section 4.3 — NYC drought zone values
# ---------------------------------------------------------------------------

def _load_reconstruction_contribution_metrics() -> pd.DataFrame:
    """
    Load pre-computed contribution metrics for the reconstruction.
    Falls back to computing from raw historical data if CSV is absent.
    """
    from methods.config import PERFORMANCE_METRICS_DIR
    import os
    csv_path = f"{PERFORMANCE_METRICS_DIR}/reconstruction_contribution_metrics.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if "annual_max_zone_date" in df.columns:
            df["annual_max_zone_date"] = pd.to_datetime(df["annual_max_zone_date"])
        if "annual_min_storage_date" in df.columns:
            df["annual_min_storage_date"] = pd.to_datetime(df["annual_min_storage_date"])
        return df

    # Compute on the fly using the postprocess pipeline
    from methods.postprocess import calculate_contribution_analysis_metrics
    hist_data = load_and_process_historical_models("reconstruction")

    # Flatten historical_data structure into a pseudo-Data object
    import pywrdrb
    pseudo = pywrdrb.Data()
    pseudo.res_level = {"reconstruction": hist_data["res_level"]["reconstruction"]}
    pseudo.res_storage = {"reconstruction": hist_data["res_storage"]["reconstruction"]}
    pseudo.inflow = {"reconstruction": hist_data["inflow"]["reconstruction"]}
    pseudo.ibt_diversions = {"reconstruction": hist_data["ibt_diversions"]["reconstruction"]}
    pseudo.ibt_demands = {"reconstruction": hist_data["ibt_demands"]["reconstruction"]}
    pseudo.contribution = {"reconstruction": hist_data["contribution"]["reconstruction"]}

    realizations = list(hist_data["major_flow"]["reconstruction"].keys())
    metrics_df = calculate_contribution_analysis_metrics(pseudo, "reconstruction", realizations)
    return metrics_df


def section_43(out: dict) -> dict:
    print("Section 4.3: NYC drought-zone values...")

    # --- Seasonal timing (panel 6a) ---
    for key, ens_id in ENS_IDS.items():
        try:
            zp = load_zone_probabilities(ens_id, period="weekly")
            if zp is None:
                raise ValueError(f"zone_probabilities is None for {ens_id}")

            # Identify drought zone columns
            if "any_drought" in zp.columns:
                any_drought = zp["any_drought"]
            else:
                # Try various column naming conventions
                watch_cols = [c for c in zp.columns
                              if any(x in str(c).lower() for x in ["watch", "zone_4", "_4"])]
                warn_cols  = [c for c in zp.columns
                              if any(x in str(c).lower() for x in ["warning", "zone_5", "_5"])]
                emrg_cols  = [c for c in zp.columns
                              if any(x in str(c).lower() for x in ["emergency", "zone_6", "_6"])]
                # Check for simple numeric zone column names
                if not (watch_cols or warn_cols or emrg_cols):
                    watch_cols = [c for c in zp.columns if str(c) == "4"]
                    warn_cols  = [c for c in zp.columns if str(c) == "5"]
                    emrg_cols  = [c for c in zp.columns if str(c) == "6"]
                all_drought_cols = watch_cols + warn_cols + emrg_cols
                if all_drought_cols:
                    any_drought = zp[all_drought_cols].sum(axis=1)
                else:
                    raise ValueError(f"Cannot identify drought zone columns in {list(zp.columns)}")

            # Ensure index is week-of-year integer 1..52
            any_drought = any_drought.copy()
            any_drought.index = pd.to_numeric(any_drought.index, errors="coerce")
            any_drought = any_drought.dropna()
            any_drought = any_drought.sort_index()

            smoothed = any_drought.rolling(3, center=True, min_periods=1).mean()
            peak_frac = float(smoothed.max())
            peak_week = int(smoothed.idxmax())
            out[f"{key}_peak_frac"] = peak_frac
            out[f"{key}_peak_week"] = peak_week
            out[f"{key}_peak_doy"]  = peak_week * 7 - 3  # approximate mid-week DOY
        except Exception as exc:
            _log_error(f"{key}_peak_frac/week", exc)
            for k in [f"{key}_peak_frac", f"{key}_peak_week", f"{key}_peak_doy"]:
                out.setdefault(k, np.nan)

    # --- Zone-years per realization (panel 6b) ---
    for key, ens_id in ENS_IDS.items():
        try:
            cm = load_contribution_metrics(ens_id)
            unique_reals = cm["realization_id"].unique()
            for zone_name, zone_code in [("watch", 4), ("warning", 5), ("emergency", 6)]:
                counts = (cm[cm["annual_max_zone"] == zone_code]
                           .groupby("realization_id")
                           .size()
                           .reindex(unique_reals, fill_value=0))
                out[f"{key}_median_{zone_name}_years"] = float(counts.median())
                out[f"{key}_q25_{zone_name}_years"]    = float(counts.quantile(0.25))
                out[f"{key}_q75_{zone_name}_years"]    = float(counts.quantile(0.75))
                out[f"{key}_iqr_{zone_name}"]          = float(
                    counts.quantile(0.75) - counts.quantile(0.25))
                out[f"{key}_q99_{zone_name}_years"]    = float(counts.quantile(0.99))
                out[f"{key}_max_{zone_name}_years"]    = int(counts.max())
        except Exception as exc:
            _log_error(f"{key}_zone_years", exc)
            for zone_name in ["watch", "warning", "emergency"]:
                for suffix in ["median", "q25", "q75", "iqr", "q99", "max"]:
                    k = (f"{key}_{suffix}_{zone_name}_years"
                         if suffix != "iqr" else f"{key}_iqr_{zone_name}")
                    out.setdefault(k, np.nan)

    # --- Climate shifts in zone-year medians ---
    for z in ["watch", "warning"]:
        for src in ["wwss", "wwds"]:
            k_shift = f"{src}_median_{z}_shift"
            k_src = f"{src}_median_{z}_years"
            k_ssb = f"ssb_median_{z}_years"
            _safe(out, k_shift,
                  lambda a=k_src, b=k_ssb: out.get(a, np.nan) - out.get(b, np.nan))

    # --- Reconstruction zone frequencies (panel 6b triangles) ---
    try:
        rec_cm = _load_reconstruction_contribution_metrics()
        for zone_name, zone_code in [("watch", 4), ("warning", 5), ("emergency", 6)]:
            raw = int((rec_cm["annual_max_zone"] == zone_code).sum())
            out[f"hist_{zone_name}_years_raw"]    = raw
            out[f"hist_{zone_name}_years_scaled"] = float(raw * 70 / RECONSTRUCTION_N_YEARS)
    except Exception as exc:
        _log_error("hist_zone_years_raw/scaled", exc)
        for zone_name in ["watch", "warning", "emergency"]:
            out.setdefault(f"hist_{zone_name}_years_raw", np.nan)
            out.setdefault(f"hist_{zone_name}_years_scaled", np.nan)

    # --- Event durations per zone (panel 6c) ---
    # Uses zone_duration_metrics.calculate_drought_zone_events on res_level data
    for key, ens_id in ENS_IDS.items():
        try:
            import pywrdrb
            fname = f"{OUTPUT_DIR}/{ens_id}_with_postprocessing.hdf5"
            data_dur = pywrdrb.Data()
            data_dur.load_from_export(fname, results_sets=["res_level"])
            realizations_dur = sorted(data_dur.res_level[ens_id].keys())

            zone_events_by_max: dict[int, list[float]] = {4: [], 5: [], 6: []}
            for r in realizations_dur:
                zone_series = data_dur.res_level[ens_id][r]["nyc"]
                episodes = calculate_drought_zone_events(zone_series, min_end_days=7)
                for ep in episodes:
                    mz = int(ep.get("max_zone", 0))
                    if mz in zone_events_by_max:
                        zone_events_by_max[mz].append(ep["duration_days"] / 30.44)

            zone_name_map = {4: "watch", 5: "warning", 6: "emergency"}
            all_duration_medians: list[float] = []
            for zone_code, zone_name in zone_name_map.items():
                durs = zone_events_by_max[zone_code]
                if not durs:
                    for suf in ["duration_median", "duration_q75",
                                "duration_p90", "duration_tail"]:
                        out[f"{key}_{zone_name}_{suf}"] = np.nan
                    continue
                durs_arr = np.array(durs)
                out[f"{key}_{zone_name}_duration_median"] = float(np.median(durs_arr))
                out[f"{key}_{zone_name}_duration_q75"]    = float(np.percentile(durs_arr, 75))
                out[f"{key}_{zone_name}_duration_p90"]    = float(np.percentile(durs_arr, 90))
                out[f"{key}_{zone_name}_duration_tail"]   = float(durs_arr.max())
                all_duration_medians.append(float(np.median(durs_arr)))
        except Exception as exc:
            _log_error(f"{key}_zone_duration", exc)
            for zone_name in ["watch", "warning", "emergency"]:
                for suf in ["duration_median", "duration_q75",
                            "duration_p90", "duration_tail"]:
                    out.setdefault(f"{key}_{zone_name}_{suf}", np.nan)

    # --- max_duration_median_shift_months ---
    try:
        shifts: list[float] = []
        for zone_name in ["watch", "warning", "emergency"]:
            ssb_med = out.get(f"ssb_{zone_name}_duration_median", np.nan)
            if np.isnan(ssb_med):
                continue
            for ens_key in ["wwds", "wwss"]:
                ens_med = out.get(f"{ens_key}_{zone_name}_duration_median", np.nan)
                if not np.isnan(ens_med):
                    shifts.append(abs(ens_med - ssb_med))
        out["max_duration_median_shift_months"] = float(max(shifts)) if shifts else np.nan
    except Exception as exc:
        _log_error("max_duration_median_shift_months", exc)
        out.setdefault("max_duration_median_shift_months", np.nan)

    # --- hist_emergency_duration_max ---
    try:
        import pywrdrb
        fname_rec = f"{OUTPUT_DIR}/reconstruction.hdf5"
        # reconstruction.hdf5 is a raw pywr output (not Data.export() schema),
        # so it must be loaded via load_output().
        data_rec_dur = pywrdrb.Data(results_sets=["res_level"], print_status=False)
        data_rec_dur.load_output(output_filenames=[fname_rec])
        rec_realizations = sorted(data_rec_dur.res_level["reconstruction"].keys())
        emrg_durs: list[float] = []
        for r in rec_realizations:
            zs = data_rec_dur.res_level["reconstruction"][r]["nyc"]
            eps = calculate_drought_zone_events(zs, min_end_days=7)
            for ep in eps:
                if ep.get("max_zone", 0) == 6:
                    emrg_durs.append(ep["duration_days"] / 30.44)
        out["hist_emergency_duration_max"] = float(max(emrg_durs)) if emrg_durs else np.nan
    except Exception as exc:
        _log_error("hist_emergency_duration_max", exc)
        out.setdefault("hist_emergency_duration_max", np.nan)

    return out


# ---------------------------------------------------------------------------
# Section 4.4 — Figures 7 and 8
# ---------------------------------------------------------------------------

def section_44(out: dict) -> dict:
    print("Section 4.4: contribution/diversion KDEs and Montague share...")

    # --- Stressed vs non-stressed fractions ---
    for key, ens_id in ENS_IDS.items():
        try:
            cm = load_contribution_metrics(ens_id)
            out[f"{key}_nonstressed_frac"]   = float((cm["annual_max_zone"] <= 3).mean())
            out[f"{key}_stressed_frac"]      = float((cm["annual_max_zone"] >= 4).mean())
            out[f"{key}_watch_only_frac"]    = float((cm["annual_max_zone"] == 4).mean())
            out[f"{key}_warning_only_frac"]  = float((cm["annual_max_zone"] == 5).mean())
            out[f"{key}_emergency_only_frac"] = float((cm["annual_max_zone"] == 6).mean())
        except Exception as exc:
            _log_error(f"{key}_stressed_frac", exc)
            for k in [f"{key}_nonstressed_frac", f"{key}_stressed_frac",
                      f"{key}_watch_only_frac", f"{key}_warning_only_frac",
                      f"{key}_emergency_only_frac"]:
                out.setdefault(k, np.nan)

    # --- KDE peaks for contribution and diversion ratios ---
    for key, ens_id in ENS_IDS.items():
        try:
            cm = load_contribution_metrics(ens_id)
            for zone_grp_name, zone_cond in [
                ("nonstressed", cm["annual_max_zone"] <= 3),
                ("stressed",    cm["annual_max_zone"] >= 4),
            ]:
                sub = cm[zone_cond]
                for ratio_type in ["contribution", "diversion"]:
                    for window_name, W in [("3mo", 90), ("9mo", 270)]:
                        col = f"{ratio_type}_ratio_{W}d"
                        if col not in sub.columns:
                            for k in [f"{key}_{zone_grp_name}_{ratio_type}_peak_{window_name}",
                                      f"{key}_{zone_grp_name}_{ratio_type}_p50_{window_name}",
                                      f"{key}_{zone_grp_name}_{ratio_type}_p95_{window_name}",
                                      f"{key}_{zone_grp_name}_{ratio_type}_mean_{window_name}"]:
                                out.setdefault(k, np.nan)
                            continue
                        vals = sub[col].dropna().values
                        if len(vals) < 5:
                            for k in [f"{key}_{zone_grp_name}_{ratio_type}_peak_{window_name}",
                                      f"{key}_{zone_grp_name}_{ratio_type}_p50_{window_name}",
                                      f"{key}_{zone_grp_name}_{ratio_type}_p95_{window_name}",
                                      f"{key}_{zone_grp_name}_{ratio_type}_mean_{window_name}"]:
                                out.setdefault(k, np.nan)
                            continue
                        try:
                            kde = gaussian_kde(vals)
                            grid = np.linspace(vals.min(), vals.max(), 2000)
                            peak = float(grid[kde(grid).argmax()])
                        except Exception:
                            peak = float(np.median(vals))
                        out[f"{key}_{zone_grp_name}_{ratio_type}_peak_{window_name}"] = peak
                        out[f"{key}_{zone_grp_name}_{ratio_type}_p50_{window_name}"]  = float(np.percentile(vals, 50))
                        out[f"{key}_{zone_grp_name}_{ratio_type}_p95_{window_name}"]  = float(np.percentile(vals, 95))
                        out[f"{key}_{zone_grp_name}_{ratio_type}_mean_{window_name}"] = float(np.mean(vals))
        except Exception as exc:
            _log_error(f"{key}_kde_peaks", exc)

    # --- Text aliases (SSB) ---
    alias_map = {
        "nonstressed_diversion_mode_3mo":  "ssb_nonstressed_diversion_peak_3mo",
        "nonstressed_diversion_mode_9mo":  "ssb_nonstressed_diversion_peak_9mo",
        "stressed_contribution_peak_3mo":  "ssb_stressed_contribution_peak_3mo",
        "stressed_diversion_peak_3mo":     "ssb_stressed_diversion_peak_3mo",
        "stressed_contribution_peak_9mo":  "ssb_stressed_contribution_peak_9mo",
        "stressed_diversion_peak_9mo":     "ssb_stressed_diversion_peak_9mo",
    }
    for alias, src in alias_map.items():
        out[alias] = out.get(src, np.nan)

    # --- 1960s contribution/diversion ratios from reconstruction ---
    try:
        rec_cm = _load_reconstruction_contribution_metrics()
        # Find year in reconstruction where annual_max_zone == 6 in 1960s
        mask_1960s = (rec_cm["year"].between(1964, 1967)) & (rec_cm["annual_max_zone"] == 6)
        if mask_1960s.sum() == 0:
            # Relax to any drought zone in 1960s
            mask_1960s = rec_cm["year"].between(1964, 1967)
        row_1960s = rec_cm[mask_1960s].sort_values("annual_max_zone", ascending=False).iloc[0]
        out["hist_1960s_contribution_3mo"] = float(row_1960s.get("contribution_ratio_90d", np.nan))
        out["hist_1960s_diversion_3mo"]    = float(row_1960s.get("diversion_ratio_90d", np.nan))
        out["hist_1960s_contribution_9mo"] = float(row_1960s.get("contribution_ratio_270d", np.nan))
        out["hist_1960s_diversion_9mo"]    = float(row_1960s.get("diversion_ratio_270d", np.nan))
    except Exception as exc:
        _log_error("hist_1960s_contribution/diversion_3mo/9mo", exc)
        for k in ["hist_1960s_contribution_3mo", "hist_1960s_diversion_3mo",
                  "hist_1960s_contribution_9mo", "hist_1960s_diversion_9mo"]:
            out.setdefault(k, np.nan)

    # --- Figure 8: pooled daily mandated NYC share of Montague ---
    # Indexed by day-of-water-year 1..365
    try:
        import pywrdrb

        day_of_wy: dict[str, dict[int, list[float]]] = {}
        for key, ens_id in ENS_IDS.items():
            day_of_wy[key] = {d: [] for d in range(1, 366)}
            fname = f"{OUTPUT_DIR}/{ens_id}_with_postprocessing.hdf5"
            data_f8 = pywrdrb.Data()
            data_f8.load_from_export(
                fname,
                results_sets=["contribution", "major_flow"]
            )
            realizations_f8 = sorted(data_f8.major_flow[ens_id].keys())

            for r in realizations_f8:
                # 'contribution' is the per-realization sum of
                # mrf_montagueTrenton_<res> across NYC reservoirs, written by
                # 04_postprocess_data_mpi.py as a single-column DataFrame.
                contrib_df = data_f8.contribution[ens_id][r]
                mont_flow = data_f8.major_flow[ens_id][r]["delMontague"]

                if "mrf_montagueTrenton_nyc" not in contrib_df.columns:
                    break  # no mandated release column
                nyc_mandate = contrib_df["mrf_montagueTrenton_nyc"]

                # Align to common index
                common_idx_f8 = nyc_mandate.index.intersection(mont_flow.index)
                nyc_mandate = nyc_mandate.reindex(common_idx_f8)
                mont_flow_r = mont_flow.reindex(common_idx_f8).replace(0, np.nan)
                ratio = 100.0 * nyc_mandate / mont_flow_r

                wy_doy_r = vectorized_water_year_doy(ratio.index)
                for d in range(1, 366):
                    mask_d = wy_doy_r == d
                    vals_d = ratio[mask_d].dropna().values
                    if len(vals_d) > 0:
                        day_of_wy[key][d].extend(vals_d.tolist())

        # Summarize SSB
        ssb_doy = day_of_wy.get("ssb", {})
        if ssb_doy and any(len(v) > 0 for v in ssb_doy.values()):
            # Jun 1 = day 1; Jun 1 through Jul 15 ≈ days 1-45
            medians = {d: np.nanmedian(ssb_doy[d]) if ssb_doy[d] else np.nan
                       for d in range(1, 366)}
            q25_d  = {d: np.nanpercentile(ssb_doy[d], 25) if ssb_doy[d] else np.nan
                      for d in range(1, 366)}
            q75_d  = {d: np.nanpercentile(ssb_doy[d], 75) if ssb_doy[d] else np.nan
                      for d in range(1, 366)}
            q99_d  = {d: np.nanpercentile(ssb_doy[d], 99) if ssb_doy[d] else np.nan
                      for d in range(1, 366)}

            jun_jul_range = range(1, 46)
            out["ssb_median_share_JunJul"] = float(
                max(medians[d] for d in jun_jul_range if not np.isnan(medians[d])))
            out["ssb_q99_share_JunJul"] = float(
                max(q99_d[d] for d in jun_jul_range if not np.isnan(q99_d[d])))

            peak_day = max(range(1, 366), key=lambda d: medians.get(d, -np.inf))
            out["ssb_median_share_peak"]      = float(medians[peak_day])
            out["ssb_median_share_peak_day"]  = int(peak_day)
            out["ssb_iqr_share_peak_low"]     = float(q25_d[peak_day])
            out["ssb_iqr_share_peak_high"]    = float(q75_d[peak_day])
            out["ssb_q99_share_peak"]         = float(q99_d[peak_day])
        else:
            for k in ["ssb_median_share_JunJul", "ssb_q99_share_JunJul",
                      "ssb_median_share_peak", "ssb_median_share_peak_day",
                      "ssb_iqr_share_peak_low", "ssb_iqr_share_peak_high",
                      "ssb_q99_share_peak"]:
                out.setdefault(k, np.nan)

        # --- Figure 8 climate shift quantile statistics ---
        quantile_levels = np.arange(0.00, 1.01, 0.01)
        ssb_doy_ok = ssb_doy and any(len(v) > 0 for v in ssb_doy.values())
        for ens_key in ["wwds", "wwss"]:
            ens_doy = day_of_wy.get(ens_key, {})
            ens_doy_ok = ens_doy and any(len(v) > 0 for v in ens_doy.values())
            if not ssb_doy_ok or not ens_doy_ok:
                out[f"{ens_key}_median_shift_sep"] = np.nan
                out[f"{ens_key}_q99_shift_sep"] = np.nan
                continue

            # September: June 1 = day 1, Sep 1 ≈ day 93, Sep 30 ≈ day 122
            sep_range = range(93, 123)
            sep_median_shifts: list[float] = []
            sep_q99_shifts: list[float] = []
            for d in sep_range:
                if not ssb_doy.get(d) or not ens_doy.get(d):
                    continue
                ssb_q = np.nanpercentile(ssb_doy[d], quantile_levels * 100)
                ens_q = np.nanpercentile(ens_doy[d], quantile_levels * 100)
                q_shifts = ens_q - ssb_q
                sep_median_shifts.append(float(np.nanmedian(q_shifts)))
                sep_q99_shifts.append(float(np.nanpercentile(q_shifts, 99)))

            if sep_median_shifts:
                if ens_key == "wwss":
                    # signed min (most negative)
                    out[f"{ens_key}_median_shift_sep"] = float(min(sep_median_shifts))
                    out[f"{ens_key}_q99_shift_sep"]    = float(min(sep_q99_shifts))
                else:
                    out[f"{ens_key}_median_shift_sep"] = float(max(sep_median_shifts))
                    out[f"{ens_key}_q99_shift_sep"]    = float(max(sep_q99_shifts))
            else:
                out[f"{ens_key}_median_shift_sep"] = np.nan
                out[f"{ens_key}_q99_shift_sep"]    = np.nan

    except Exception as exc:
        _log_error("figure8_pooled_share", exc)
        for k in ["ssb_median_share_JunJul", "ssb_q99_share_JunJul",
                  "ssb_median_share_peak", "ssb_median_share_peak_day",
                  "ssb_iqr_share_peak_low", "ssb_iqr_share_peak_high",
                  "ssb_q99_share_peak",
                  "wwds_median_shift_sep", "wwds_q99_shift_sep",
                  "wwss_median_shift_sep", "wwss_q99_shift_sep"]:
            out.setdefault(k, np.nan)

    return out


# ---------------------------------------------------------------------------
# Section 4.5 — Hazard-to-outcome mapping (Figure 9)
# ---------------------------------------------------------------------------

def section_45(out: dict) -> dict:
    print("Section 4.5: Figure 9 hazard-outcome mapping...")

    severity_edges  = np.linspace(1.0, 4.5, 17)   # 16 bins
    magnitude_edges = np.logspace(0, 2, 17)         # 16 bins, 1..100

    # Store per-ensemble arrays for focal region computation
    counts_by_ens:       dict[str, np.ndarray] = {}
    emerg_avoid_by_ens:  dict[str, np.ndarray] = {}
    min_storage_by_ens:  dict[str, np.ndarray] = {}

    for key, ens_id in ENS_IDS.items():
        try:
            ev = load_event_metrics(ens_id, ssi_window=3)

            # Bin each event
            ev = ev.copy()
            ev["sev_bin"] = np.clip(
                np.digitize(ev["severity"].values, severity_edges) - 1, 0, 15)
            ev["mag_bin"] = np.clip(
                np.digitize(ev["magnitude"].values, magnitude_edges) - 1, 0, 15)

            counts     = np.zeros((16, 16), dtype=int)
            emrg_avoid = np.full((16, 16), np.nan)
            min_stor   = np.full((16, 16), np.nan)

            for i in range(16):
                for j in range(16):
                    mask_ij = (ev["sev_bin"] == i) & (ev["mag_bin"] == j)
                    cnt = int(mask_ij.sum())
                    counts[i, j] = cnt
                    if cnt >= 5:
                        sub_ij = ev[mask_ij]
                        # Emergency avoidance fraction (zone at min != Emergency)
                        if "ffmp_zone_at_min" in sub_ij.columns:
                            emrg_avoid[i, j] = float(
                                (sub_ij["ffmp_zone_at_min"].str.lower() != "emergency").mean()
                            )
                        if "min_storage" in sub_ij.columns:
                            min_stor[i, j] = float(sub_ij["min_storage"].min())

            counts_by_ens[key]      = counts
            emerg_avoid_by_ens[key] = emrg_avoid
            min_storage_by_ens[key] = min_stor

            out[f"{key}_max_bin_count"]       = int(counts.max())
            out[f"{key}_populated_bin_count"] = int((counts >= 5).sum())

            # Total ensemble-years for exceedance rate denominator
            n_reals_ev = ev["realization_id"].nunique() if "realization_id" in ev.columns else 2000
            total_ens_years = n_reals_ev * N_YEARS
            exc_rate = counts / total_ens_years

            # High-frequency bins (top 5 by count in SSB)
            if key == "ssb":
                flat_idx = np.argsort(counts.ravel())[::-1][:5]
                top5_i = flat_idx // 16
                top5_j = flat_idx % 16
                sev_lo = float(severity_edges[top5_i.min()])
                sev_hi = float(severity_edges[min(top5_i.max() + 1, 16)])
                mag_hi = float(magnitude_edges[min(top5_j.max() + 1, 16)])
                out["ssb_high_freq_sev_range"] = f"[{sev_lo:.2f}, {sev_hi:.2f}]"
                out["ssb_high_freq_mag_cap"]   = float(mag_hi)

                # Emergency-avoid band: bins with emrg_avoid < 0.7
                low_ea = (emrg_avoid < 0.7) & (counts >= 5)
                if low_ea.any():
                    li = np.where(low_ea)[0]
                    lj = np.where(low_ea)[1]
                    out["ssb_emergency_sev_band"] = (
                        f"[{severity_edges[li.min()]:.2f}, {severity_edges[min(li.max()+1,16)]:.2f}]")
                    out["ssb_emergency_mag_band"] = (
                        f"[{magnitude_edges[lj.min()]:.2f}, {magnitude_edges[min(lj.max()+1,16)]:.2f}]")
                else:
                    out["ssb_emergency_sev_band"] = "none"
                    out["ssb_emergency_mag_band"] = "none"

            elif key == "wwds":
                # Expanded severity/magnitude band: exceedance rate in WWDS >= 1e-4
                # but in SSB < 1e-4
                if "ssb" in counts_by_ens:
                    ssb_exc = counts_by_ens["ssb"] / (
                        (ev["realization_id"].nunique() if "realization_id" in ev.columns else 2000)
                        * N_YEARS)
                    wwds_exc = counts / total_ens_years
                    expanded = (wwds_exc >= 1e-4) & (ssb_exc < 1e-4)
                    if expanded.any():
                        ei = np.where(expanded)[0]
                        ej = np.where(expanded)[1]
                        out["wwds_expanded_sev_band"] = (
                            f"[{severity_edges[ei.min()]:.2f}, {severity_edges[min(ei.max()+1,16)]:.2f}]")
                        out["wwds_expanded_mag_band"] = (
                            f"[{magnitude_edges[ej.min()]:.2f}, {magnitude_edges[min(ej.max()+1,16)]:.2f}]")
                    else:
                        out["wwds_expanded_sev_band"] = "none"
                        out["wwds_expanded_mag_band"] = "none"

                if emrg_avoid is not None:
                    valid_ea = emrg_avoid[(counts >= 5) & ~np.isnan(emrg_avoid)]
                    out["wwds_emergency_frac_floor"] = float(valid_ea.min()) if len(valid_ea) > 0 else np.nan
                else:
                    out["wwds_emergency_frac_floor"] = np.nan

        except Exception as exc:
            _log_error(f"{key}_figure9", exc)
            for k in [f"{key}_max_bin_count", f"{key}_populated_bin_count"]:
                out.setdefault(k, np.nan)
            if key == "ssb":
                for k in ["ssb_high_freq_sev_range", "ssb_high_freq_mag_cap",
                          "ssb_emergency_sev_band", "ssb_emergency_mag_band"]:
                    out.setdefault(k, np.nan)
            elif key == "wwds":
                for k in ["wwds_expanded_sev_band", "wwds_expanded_mag_band",
                          "wwds_emergency_frac_floor"]:
                    out.setdefault(k, np.nan)

    # --- Focal region: bins meeting all three criteria ---
    try:
        if len(counts_by_ens) == 3:
            # Need per-ensemble total_ens_years; approximate at 2000 * N_YEARS
            total_ens_years_approx = 2000 * N_YEARS

            # Criterion 1: exc_rate >= 1e-4 in ALL three ensembles
            all_populated = np.ones((16, 16), dtype=bool)
            for key in ["ssb", "wwds", "wwss"]:
                exc = counts_by_ens[key] / total_ens_years_approx
                all_populated &= (exc >= 1e-4)

            # Criterion 2: emrg_avoid < 0.95 in ALL three ensembles
            all_low_avoid = np.ones((16, 16), dtype=bool)
            for key in ["ssb", "wwds", "wwss"]:
                ea = emerg_avoid_by_ens[key]
                # Bins with nan (< 5 events) do not meet criterion
                valid = (~np.isnan(ea)) & (ea < 0.95)
                all_low_avoid &= valid

            # Criterion 3: worst min_storage < 15% in AT LEAST ONE ensemble
            any_low_storage = np.zeros((16, 16), dtype=bool)
            for key in ["ssb", "wwds", "wwss"]:
                ms = min_storage_by_ens[key]
                any_low_storage |= (~np.isnan(ms)) & (ms < 15.0)

            focal = all_populated & all_low_avoid & any_low_storage

            out["focal_region_bin_count"] = int(focal.sum())
            if focal.any():
                fi = np.where(focal)[0]
                fj = np.where(focal)[1]
                out["focal_region_sev_min"] = float(severity_edges[fi.min()])
                out["focal_region_sev_max"] = float(severity_edges[min(fi.max() + 1, 16)])
                out["focal_region_mag_min"] = float(magnitude_edges[fj.min()])
                out["focal_region_mag_max"] = float(magnitude_edges[min(fj.max() + 1, 16)])
            else:
                out["focal_region_sev_min"] = np.nan
                out["focal_region_sev_max"] = np.nan
                out["focal_region_mag_min"] = np.nan
                out["focal_region_mag_max"] = np.nan

            # Event counts in focal region per ensemble
            focal_grid = focal  # boolean 16x16
            for key, ens_id in ENS_IDS.items():
                try:
                    ev2 = load_event_metrics(ens_id, ssi_window=3)
                    ev2 = ev2.copy()
                    ev2["sev_bin"] = np.clip(
                        np.digitize(ev2["severity"].values, severity_edges) - 1, 0, 15)
                    ev2["mag_bin"] = np.clip(
                        np.digitize(ev2["magnitude"].values, magnitude_edges) - 1, 0, 15)
                    in_focal = np.array([
                        bool(focal_grid[int(row["sev_bin"]), int(row["mag_bin"])])
                        for _, row in ev2.iterrows()
                    ])
                    out[f"focal_region_total_events_{key}"] = int(in_focal.sum())
                except Exception as exc2:
                    _log_error(f"focal_region_total_events_{key}", exc2)
                    out[f"focal_region_total_events_{key}"] = np.nan

        else:
            for k in ["focal_region_bin_count", "focal_region_sev_min",
                      "focal_region_sev_max", "focal_region_mag_min",
                      "focal_region_mag_max"]:
                out.setdefault(k, np.nan)
            for key in ["ssb", "wwds", "wwss"]:
                out.setdefault(f"focal_region_total_events_{key}", np.nan)
    except Exception as exc:
        _log_error("focal_region", exc)
        for k in ["focal_region_bin_count", "focal_region_sev_min",
                  "focal_region_sev_max", "focal_region_mag_min",
                  "focal_region_mag_max"]:
            out.setdefault(k, np.nan)
        for key in ["ssb", "wwds", "wwss"]:
            out.setdefault(f"focal_region_total_events_{key}", np.nan)

    return out


# ---------------------------------------------------------------------------
# Section 4.6 — Focal drought event selection
# ---------------------------------------------------------------------------

def section_46(out: dict, focal_selections: dict) -> tuple[dict, dict]:
    print("Section 4.6: focal drought events...")

    anchor_sev = out.get("hist_1960s_ssi3_severity", np.nan)
    anchor_mag = out.get("hist_1960s_ssi3_magnitude", np.nan)

    severity_edges  = np.linspace(1.0, 4.5, 17)
    magnitude_edges = np.logspace(0, 2, 17)

    # Reconstruct focal_grid from out if available
    fr_sev_min = out.get("focal_region_sev_min", np.nan)
    fr_sev_max = out.get("focal_region_sev_max", np.nan)
    fr_mag_min = out.get("focal_region_mag_min", np.nan)
    fr_mag_max = out.get("focal_region_mag_max", np.nan)

    for key, ens_id in ENS_IDS.items():
        try:
            ev = load_event_metrics(ens_id, ssi_window=3)

            # Filter to focal region if bounds are available
            if not any(np.isnan(v) for v in [fr_sev_min, fr_sev_max,
                                              fr_mag_min, fr_mag_max]):
                ev_focal = ev[
                    (ev["severity"] >= fr_sev_min) & (ev["severity"] <= fr_sev_max) &
                    (ev["magnitude"] >= fr_mag_min) & (ev["magnitude"] <= fr_mag_max)
                ].copy()
            else:
                ev_focal = ev.copy()

            if ev_focal.empty:
                ev_focal = ev.copy()

            # Euclidean distance in (severity, log10(magnitude)) from 1960s anchor
            if not np.isnan(anchor_sev) and not np.isnan(anchor_mag):
                ev_focal = ev_focal.copy()
                ev_focal["dist"] = np.sqrt(
                    (ev_focal["severity"] - anchor_sev) ** 2
                    + (np.log10(ev_focal["magnitude"].clip(lower=1e-9)) -
                       np.log10(max(anchor_mag, 1e-9))) ** 2
                )
                # Sort by distance ascending, tie-break by min_storage ascending
                sort_cols = ["dist"]
                if "min_storage" in ev_focal.columns:
                    sort_cols.append("min_storage")
                ev_sorted = ev_focal.sort_values(sort_cols)
            else:
                ev_sorted = ev_focal

            if ev_sorted.empty:
                continue

            best = ev_sorted.iloc[0]

            def _col(df, *names):
                for n in names:
                    if n in df.index:
                        return df[n]
                return None

            focal_selections[key] = {
                "realization_id":    int(best["realization_id"]) if "realization_id" in best.index else None,
                "event_start":       str(best.get("start", best.get("event_start", None))),
                "event_end":         str(best.get("end", best.get("event_end", None))),
                "max_severity_date": str(best.get("max_severity_date", None)),
                "min_storage_date":  str(best.get("min_storage_date", None)),
                "severity":          float(best["severity"]),
                "magnitude":         float(best["magnitude"]),
                "duration_days":     int(best.get("duration_days", 0)),
                "ffmp_zone_at_min":  str(best.get("ffmp_zone_at_min", "")),
                "min_storage":       float(best.get("min_storage", np.nan)),
                "distance_to_anchor": float(best.get("dist", np.nan)),
            }

            # --- Focal event dynamics values ---
            # Load daily timeseries for the selected realization
            r_sel = focal_selections[key]["realization_id"]
            event_start = pd.Timestamp(focal_selections[key]["event_start"])
            event_end   = pd.Timestamp(focal_selections[key]["event_end"])

            if r_sel is None:
                continue

            try:
                import pywrdrb
                fname = f"{OUTPUT_DIR}/{ens_id}_with_postprocessing.hdf5"
                data_dyn = pywrdrb.Data()
                data_dyn.load_from_export(
                    fname,
                    results_sets=["res_storage", "res_level", "contribution",
                                  "major_flow", "ibt_diversions"],
                    realizations=[r_sel],
                )

                # Storage and zone series
                stor_r = (data_dyn.res_storage[ens_id][r_sel][NYC_RESERVOIRS].sum(axis=1)
                          / NYC_TOTAL_CAPACITY * 100.0)
                zone_r = data_dyn.res_level[ens_id][r_sel]["nyc"]
                mont_r = data_dyn.major_flow[ens_id][r_sel]["delMontague"]

                # Window: 2 months before event to 3 months after
                window_start = event_start - pd.Timedelta(days=60)
                window_end   = event_end   + pd.Timedelta(days=90)
                stor_w = stor_r.loc[window_start:window_end]
                zone_w = zone_r.loc[window_start:window_end]

                # Depletion months: from storage peak before event start to Emergency entry
                pre_event = stor_r.loc[window_start:event_start]
                storage_peak_date = pre_event.idxmax() if len(pre_event) > 0 else event_start
                emrg_dates = zone_r.loc[event_start:event_end][zone_r.loc[event_start:event_end] >= 6]
                if len(emrg_dates) > 0:
                    emrg_entry = emrg_dates.index[0]
                    dep_months = (emrg_entry - storage_peak_date).days / 30.44
                    out[f"focal_{key}_depletion_months"] = float(dep_months)
                    out[f"focal_{key}_emergency_entry_month"] = int(emrg_entry.month)
                else:
                    out[f"focal_{key}_depletion_months"] = np.nan
                    out[f"focal_{key}_emergency_entry_month"] = np.nan

                # Warning entry month
                warn_dates = zone_r.loc[event_start:event_end][zone_r.loc[event_start:event_end] >= 5]
                if len(warn_dates) > 0:
                    out[f"focal_{key}_warning_entry_month"] = int(warn_dates.index[0].month)
                else:
                    out[f"focal_{key}_warning_entry_month"] = np.nan

                # min_storage during event
                stor_ev = stor_r.loc[event_start:event_end]
                out[f"focal_{key}_min_storage"] = float(stor_ev.min()) if len(stor_ev) > 0 else np.nan

                # Mandate dominant days: NYC mandate > natural Montague flow.
                # 'contribution' holds the pre-aggregated NYC mandate column
                # (sum of mrf_montagueTrenton_<res> across NYC reservoirs).
                contrib_df = data_dyn.contribution[ens_id][r_sel]
                if "mrf_montagueTrenton_nyc" in contrib_df.columns:
                    mandate = contrib_df["mrf_montagueTrenton_nyc"]
                    mandate_ev = mandate.reindex(stor_ev.index, fill_value=0)
                    mont_ev = mont_r.reindex(stor_ev.index, fill_value=np.nan)
                    natural_flow = mont_ev - mandate_ev
                    dominant_days = int((mandate_ev > natural_flow.clip(lower=0)).sum())
                    out[f"focal_{key}_mandate_dominant_days"] = dominant_days
                else:
                    out[f"focal_{key}_mandate_dominant_days"] = np.nan

                # Recovery months: Emergency exit to Normal return
                post_event = zone_r.loc[event_end:]
                emrg_exit_dates = post_event[post_event < 6]
                normal_return_dates = post_event[post_event <= 3]
                if len(emrg_exit_dates) > 0 and len(normal_return_dates) > 0:
                    emrg_exit = emrg_exit_dates.index[0]
                    normal_return = normal_return_dates.index[0]
                    recovery = (normal_return - emrg_exit).days / 30.44
                    out[f"focal_{key}_recovery_months"] = float(max(0, recovery))
                else:
                    out[f"focal_{key}_recovery_months"] = np.nan

                # Event duration in months
                event_dur_months = (event_end - event_start).days / 30.44
                out[f"focal_{key}_event_duration_months"] = float(event_dur_months)

            except Exception as exc_dyn:
                _log_error(f"focal_{key}_dynamics", exc_dyn)
                for k in [f"focal_{key}_depletion_months", f"focal_{key}_warning_entry_month",
                          f"focal_{key}_emergency_entry_month",
                          f"focal_{key}_mandate_dominant_days",
                          f"focal_{key}_min_storage", f"focal_{key}_recovery_months",
                          f"focal_{key}_event_duration_months"]:
                    out.setdefault(k, np.nan)

        except Exception as exc:
            _log_error(f"focal_event_selection_{key}", exc)

    # --- Relative shift keys ---
    for k, a, b in [
        ("focal_wwds_warning_entry_diff",
         "focal_ssb_depletion_months", "focal_wwds_depletion_months"),
        ("focal_wwds_mandate_window_diff",
         "focal_wwds_mandate_dominant_days", "focal_ssb_mandate_dominant_days"),
        ("focal_wwss_warning_entry_diff",
         "focal_wwss_depletion_months", "focal_ssb_depletion_months"),
        ("focal_wwss_mandate_window_diff",
         "focal_ssb_mandate_dominant_days", "focal_wwss_mandate_dominant_days"),
    ]:
        try:
            va = out.get(a, np.nan)
            vb = out.get(b, np.nan)
            out[k] = float(va - vb) if not (np.isnan(va) or np.isnan(vb)) else np.nan
        except Exception as exc:
            _log_error(k, exc)
            out[k] = np.nan

    focal_selections["anchor"] = {
        "severity":  float(anchor_sev) if not np.isnan(anchor_sev) else None,
        "magnitude": float(anchor_mag) if not np.isnan(anchor_mag) else None,
    }

    return out, focal_selections


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _nan_safe_default(obj):
    """json.dump default serializer: convert numpy/float nan/inf to None."""
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Cannot serialize {type(obj)}")


SECTION_PREFIXES: dict[str, list[str]] = {
    "4.1": ["hist_acf_", "hist_fdc_", "max_abs_median_acf_", "summer_lf_"],
    "4.2": ["ssi3_", "ssi6_", "ssi12_", "hist_1960s_",
            "count_events_", "ssb_ex_rate_", "ssb_rp_1960",
            "ssb_median_exrate_", "ssb_q", "ssb_rp_",
            "ssb_min_count_", "ssb_max_count_",
            "wwds_median_exrate_", "wwds_q", "wwds_rp_", "wwds_count_",
            "wwds_delta_", "wwds_magnitude_",
            "wwss_median_exrate_", "wwss_q", "wwss_rp_", "wwss_count_",
            "wwss_delta_", "wwss_severity_tail_", "wwss_magnitude_tail_",
            "wwss_sev_thresh_"],
    "4.3": ["ssb_peak_", "wwds_peak_", "wwss_peak_",
            "ssb_median_watch_", "ssb_median_warning_", "ssb_median_emergency_",
            "ssb_q25_", "ssb_q75_", "ssb_iqr_", "ssb_q99_", "ssb_max_",
            "wwds_median_watch_", "wwds_median_warning_", "wwds_iqr_",
            "wwss_median_watch_", "wwss_median_warning_", "wwss_iqr_",
            "wwss_median_watch_shift", "wwss_median_warning_shift",
            "wwds_median_watch_shift", "wwds_median_warning_shift",
            "hist_watch_", "hist_warning_", "hist_emergency_",
            "ssb_watch_duration_", "ssb_warning_duration_", "ssb_emergency_duration_",
            "wwds_watch_duration_", "wwds_warning_duration_", "wwds_emergency_duration_",
            "wwss_watch_duration_", "wwss_warning_duration_", "wwss_emergency_duration_",
            "max_duration_", "hist_emergency_duration_"],
    "4.4": ["ssb_nonstressed_", "ssb_stressed_", "ssb_watch_only_",
            "ssb_warning_only_", "ssb_emergency_only_",
            "wwds_nonstressed_", "wwds_stressed_", "wwds_watch_only_",
            "wwss_nonstressed_", "wwss_stressed_", "wwss_watch_only_",
            "nonstressed_diversion_", "stressed_contribution_", "stressed_diversion_",
            "hist_1960s_contribution_", "hist_1960s_diversion_",
            "ssb_median_share_", "ssb_q99_share_", "ssb_iqr_share_",
            "wwds_median_shift_", "wwds_q99_shift_",
            "wwss_median_shift_", "wwss_q99_shift_",
            "ssb_nonstressed_contribution_", "ssb_stressed_contribution_",
            "ssb_nonstressed_diversion_", "ssb_stressed_diversion_",
            "wwds_nonstressed_", "wwds_stressed_",
            "wwss_nonstressed_", "wwss_stressed_"],
    "4.5": ["ssb_high_freq_", "ssb_emergency_sev_", "ssb_emergency_mag_",
            "ssb_max_bin_", "ssb_populated_",
            "wwds_expanded_", "wwds_emergency_frac_",
            "wwds_max_bin_", "wwds_populated_",
            "wwss_max_bin_", "wwss_populated_",
            "focal_region_"],
    "4.6": ["focal_ssb_", "focal_wwds_", "focal_wwss_"],
}


def _write_md_listings(out: dict) -> None:
    """Write alphabetical and section-grouped human-readable listings."""
    alpha_keys = sorted(out.keys())

    # Alphabetical listing
    lines = ["# Extracted manuscript values (rev1)\n\n",
             "Generated by `analysis/extract_manuscript_values.py`.\n\n"]
    for k in alpha_keys:
        v = out[k]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            lines.append(f"- `{k}` = NaN  **(MISSING)**\n")
        elif isinstance(v, float):
            lines.append(f"- `{k}` = {v:.6g}\n")
        else:
            lines.append(f"- `{k}` = {v}\n")
    (MS_OUT / "extracted_values_rev1.md").write_text("".join(lines), encoding="utf-8")

    # Section-grouped listing
    section_lines = ["# Extracted manuscript values by section (rev1)\n\n"]
    assigned: set[str] = set()
    for section, prefixes in SECTION_PREFIXES.items():
        section_keys = [k for k in alpha_keys
                        if any(k.startswith(p) or k == p.rstrip("_") for p in prefixes)]
        if not section_keys:
            continue
        section_lines.append(f"\n## Section {section}\n\n")
        for k in section_keys:
            assigned.add(k)
            v = out[k]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                section_lines.append(f"- `{k}` = NaN  **(MISSING)**\n")
            elif isinstance(v, float):
                section_lines.append(f"- `{k}` = {v:.6g}\n")
            else:
                section_lines.append(f"- `{k}` = {v}\n")

    # Uncategorised keys
    unassigned = [k for k in alpha_keys if k not in assigned]
    if unassigned:
        section_lines.append("\n## Uncategorised\n\n")
        for k in unassigned:
            v = out[k]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                section_lines.append(f"- `{k}` = NaN  **(MISSING)**\n")
            elif isinstance(v, float):
                section_lines.append(f"- `{k}` = {v:.6g}\n")
            else:
                section_lines.append(f"- `{k}` = {v}\n")

    (MS_OUT / "extracted_values_by_section.md").write_text(
        "".join(section_lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Validation against text_edits_by_section.md
# ---------------------------------------------------------------------------

def _validate_against_text_edits(out: dict, n_written: int) -> str:
    """
    Grep text_edits_by_section.md for every [value: <key>] placeholder,
    check each key is present in out.  Log missing keys.
    Returns a one-line summary string.
    """
    text_edits_path = MS / "text_edits_by_section.md"
    if not text_edits_path.exists():
        return f"Wrote {n_written} values; text_edits_by_section.md not found — skipping validation."

    text = text_edits_path.read_text(encoding="utf-8")
    pattern = re.compile(r"\[value:\s*([a-zA-Z0-9_]+)\]")
    required_keys = sorted(set(pattern.findall(text)))

    missing = [k for k in required_keys if k not in out or
               (isinstance(out[k], float) and np.isnan(out[k]))]

    if missing:
        msg = "Missing/NaN keys from text_edits:\n" + "\n".join(f"  {k}" for k in missing)
        logging.warning(msg)
        _error_log.append(msg)

    return (f"Wrote {n_written} values to {MS_OUT / 'extracted_values_rev1.json'}; "
            f"missing: {len(missing)} / {len(required_keys)} required keys.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SECTION_CHOICES = ["all", "4.1", "4.2", "4.3", "4.4", "4.5", "4.6"]


def _drop_section_keys(out: dict, section: str) -> dict:
    """Remove keys assigned to the given section so they are recomputed cleanly."""
    prefixes = SECTION_PREFIXES[section]
    return {
        k: v for k, v in out.items()
        if not any(k.startswith(p) or k == p.rstrip("_") for p in prefixes)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--section",
        choices=SECTION_CHOICES,
        default="all",
        help="Run a single Section 4.x block (default: all). When a single "
             "section is given, existing keys in extracted_values_rev1.json are "
             "preserved and only the targeted section's keys are recomputed.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(f"extract_manuscript_values.py — starting extraction (section={args.section})")
    print("=" * 70)

    out: dict = {}
    focal_selections: dict = {}

    json_path = MS_OUT / "extracted_values_rev1.json"
    focal_path = MS_OUT / "focal_event_selections.json"

    # Targeted rerun: preload prior values, then drop the section being rerun
    # so its keys get recomputed without losing the other sections' results.
    if args.section != "all" and json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            prior = json.load(f)
        out = _drop_section_keys(prior, args.section)
        print(f"Loaded {len(prior)} prior keys; recomputing section {args.section} "
              f"({len(prior) - len(out)} keys to refresh).")
        if focal_path.exists():
            with open(focal_path, encoding="utf-8") as f:
                focal_selections = json.load(f)

    if args.section in ("all", "4.1"):
        out = section_41(out)
    if args.section in ("all", "4.2"):
        out = section_42(out)
    if args.section in ("all", "4.3"):
        out = section_43(out)
    if args.section in ("all", "4.4"):
        out = section_44(out)
    if args.section in ("all", "4.5"):
        out = section_45(out)
    if args.section in ("all", "4.6"):
        out, focal_selections = section_46(out, focal_selections)

    # Write JSON (NaN → null)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=_nan_safe_default)

    with open(focal_path, "w", encoding="utf-8") as f:
        json.dump(focal_selections, f, indent=2, default=_nan_safe_default)

    # Write human-readable listings
    _write_md_listings(out)

    # Validate against text_edits placeholders
    summary = _validate_against_text_edits(out, len(out))
    print(summary)

    # Print in-memory error summary
    if _error_log:
        print(f"\n{len(_error_log)} computation error(s) logged to {LOG_FILE}:")
        for msg in _error_log[:20]:
            print(f"  {msg}")
        if len(_error_log) > 20:
            print(f"  ... and {len(_error_log) - 20} more — see {LOG_FILE}")
    else:
        print("No computation errors.")

    print(f"\nOutputs written to {MS_OUT}/")
    print(f"  extracted_values_rev1.json  ({len(out)} keys)")
    print(f"  focal_event_selections.json")
    print(f"  extracted_values_rev1.md")
    print(f"  extracted_values_by_section.md")
    if LOG_FILE.exists() and LOG_FILE.stat().st_size > 0:
        print(f"  extracted_values_rev1_errors.log")


if __name__ == "__main__":
    main()
