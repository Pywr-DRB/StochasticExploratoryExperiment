"""
SI22: Storyline drought dynamics, selected separately from each ensemble.

Main-text Fig10 (manuscript Figure 9) shows two storyline droughts drawn
from the consequential (focal) drought region using two deterministic
selection criteria:

    A.  Largest cumulative NYC diversion shortage (% of demand) over the
        event  -> ``rank_max`` on ``nyc_shortage_pct``.
    B.  Sub-annual event (<= 366 d) that reaches the FFMP Drought
        Emergency zone at its storage minimum, with the lowest combined
        NYC storage  -> ``rank_min`` on ``event_min_storage_pct``.
        The main-text B rule also carries a (2.8, 3.4] severity band;
        this SI removes both severity filters (``DROP_CRITERIA_KEYS``).

This SI applies exactly the same two criteria to every ensemble
separately, giving one 2-column figure per scenario:

    SI22_..._stationary   Events A & B from the Stationary Baseline
    SI22_..._wwds         Events A & B from WWDS (climate_adjusted_low)
    SI22_..._wwss         Events A & B from WWSS (climate_adjusted_high)

The criteria are imported from ``FIG_VERSIONS[MAIN_TEXT_VERSION]`` in
``plotting_scripts/Fig10_focal_drought_dynamics.py`` so the SI can never
drift from the main-text rules; only the ``ensemble`` (and the slot
title) is overridden per scenario. The focal region, grid edges, panel
layout, x-axis windowing, smoothing and legend are all reused from the
Fig10 module via ``_render_version``. Panel (a)'s grey support grid
shows the scenario's own event counts (Fig10 always shows the
stationary counts) and the focal-region outline is the shared
three-ensemble region.

Outputs (all under ``SI_OUTPUT_DIR``):
    SI22_storyline_droughts_ssi3_<scenario>.{png,svg,pdf}
    SI22_selected_events.csv        metadata of the six selected events
    SI22_selection_criteria.txt     the slot criteria actually applied

Usage:
    python si_scripts/SI22_scenario_storyline_droughts.py
"""

import sys
import os
import importlib.util
import warnings
warnings.filterwarnings("ignore")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import pandas as pd
import matplotlib.pyplot as plt

from methods.config import CONFIG_DIR, DATASET_CONFIGS
from methods.load import load_event_metrics
from methods.drought_analysis import fit_ssi_calculator
from methods.plotting.heatmap import compute_min_storage_grid
from methods.plotting.styles import (
    DATASET_LABELS, DATASET_LABELS_SHORT, apply_publication_style,
)


# -- import the Fig10 module (plotting_scripts/ is not a package) ------------

def _import_fig10():
    path = os.path.join(ROOT_DIR, 'plotting_scripts',
                        'Fig10_focal_drought_dynamics.py')
    spec = importlib.util.spec_from_file_location('fig10_focal', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fig10 = _import_fig10()


# -- configuration -----------------------------------------------------------

SSI_WINDOW = fig10.SSI_WINDOW
DATASETS = list(DATASET_CONFIGS.keys())

# Which Fig10 version supplies the A/B criteria. 'v3_wwssA' is the
# preferred main-text pairing: A = rank_max nyc_shortage_pct; B = rank_min
# event_min_storage_pct with sub-annual + Emergency-at-minimum filters and
# the (2.8, 3.4] severity band that excludes the region's tail-most bin.
MAIN_TEXT_VERSION = 'v3_wwssA'

# Slot-config keys removed from the imported criteria before they are
# applied. Per the SI request (2026-09-14) Event B drops BOTH severity
# filters, so B is simply the sub-annual (<= 366 d) focal event that is in
# Drought Emergency at its storage minimum with the lowest combined NYC
# storage, with no severity restriction. Set to {} to mirror the main text.
DROP_CRITERIA_KEYS = {
    'A': (),
    'B': ('filter_severity_min', 'filter_severity_max'),
}

# Scenario-neutral slot titles (the Fig10 titles name a specific ensemble).
SLOT_TITLES = {
    'A': 'Largest cumulative NYC-shortage focal drought',
    'B': 'Deepest-drawdown sub-annual Emergency focal drought',
}

# Filename suffix per scenario.
SCENARIO_SUFFIX = {
    'stationary_ensemble': 'stationary',
    'climate_adjusted_low': 'wwds',
    'climate_adjusted_high': 'wwss',
}

# Output destination. Per the SI request this sits directly under the
# config directory (outputs/<CONFIG_NAME>/SI22_...), not under figures/.
SI_OUTPUT_DIR = os.path.join(CONFIG_DIR, 'SI22_scenario_storyline_droughts')
FIG_NAME_STEM = f'SI22_storyline_droughts_ssi{SSI_WINDOW}'
EVENTS_CSV = os.path.join(SI_OUTPUT_DIR, 'SI22_selected_events.csv')
CRITERIA_TXT = os.path.join(SI_OUTPUT_DIR, 'SI22_selection_criteria.txt')


# -- helpers -----------------------------------------------------------------

def build_scenario_criteria(ensemble):
    """The main-text A/B slot configs with the ensemble swapped for
    *ensemble* and a scenario-neutral title. Every other key (selection
    method, rank column, pool filters, marker) is inherited verbatim."""
    base = fig10.FIG_VERSIONS[MAIN_TEXT_VERSION]
    out = {}
    for slot in fig10.EVENT_SLOT_ORDER:
        cfg = dict(base[slot])
        for key in DROP_CRITERIA_KEYS.get(slot, ()):
            cfg.pop(key, None)
        cfg['ensemble'] = ensemble
        cfg['title'] = SLOT_TITLES.get(slot, cfg['title'])
        out[slot] = cfg
    return out


def describe_criterion(cfg):
    method = cfg.get('selection_method', 'target_nearest')
    if method == 'target_nearest':
        txt = (f"target_nearest: z-scored nearest to severity="
               f"{cfg['target_severity']}, min_storage="
               f"{cfg['target_min_storage_pct']}%")
    else:
        txt = f"{method} on {cfg['rank_column']}"
    filters = fig10._describe_slot_filters(cfg)
    if filters:
        txt += f"; pool filters: {filters}"
    return txt


def write_criteria_file(path):
    lines = [
        f"SI22 storyline selection criteria (from Fig10 FIG_VERSIONS"
        f"['{MAIN_TEXT_VERSION}']), applied identically to each ensemble.",
        "Dropped keys per slot: " + ", ".join(
            f"{slot}={list(keys) or 'none'}"
            for slot, keys in DROP_CRITERIA_KEYS.items()),
        f"Focal region: SSI-{SSI_WINDOW}, MIN_COUNT={fig10.MIN_COUNT}, "
        f"RP<={fig10.FOCAL_RP_THRESH_YEARS} yr, "
        f"frac_thresh={fig10.FOCAL_FRAC_THRESH}, "
        f"storage_thresh={fig10.FOCAL_WORST_STORAGE_THRESH}%.",
        "",
    ]
    criteria = build_scenario_criteria('<ensemble>')
    for slot in fig10.EVENT_SLOT_ORDER:
        cfg = criteria[slot]
        lines.append(f"Event {slot} - {cfg['title']}")
        lines.append(f"    {describe_criterion(cfg)}")
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def event_record(scenario, slot, ev):
    cfg = ev['config']
    return dict(
        scenario=scenario,
        scenario_label=DATASET_LABELS_SHORT.get(scenario, scenario),
        slot=slot,
        title=cfg['title'],
        criterion=describe_criterion(cfg),
        realization_id=ev['realization_id'],
        start=ev['start'].date(),
        end=ev['end'].date(),
        duration_days=ev['duration_days'],
        severity=round(ev['severity'], 3),
        magnitude=round(ev['magnitude'], 2),
        event_min_storage_pct=round(ev['event_min_storage_pct'], 2),
        min_storage_date=ev['min_storage_date'].date(),
        nyc_shortage_pct=round(ev['nyc_shortage_pct'], 2),
    )


# -- main --------------------------------------------------------------------

def main():
    apply_publication_style()
    plt.rcParams.update({'font.size': 13, 'font.weight': 'normal'})
    os.makedirs(SI_OUTPUT_DIR, exist_ok=True)

    print(f"SI22: per-scenario storyline droughts (SSI-{SSI_WINDOW}), "
          f"criteria from Fig10 version '{MAIN_TEXT_VERSION}'")
    write_criteria_file(CRITERIA_TXT)
    print(f"  Wrote criteria: {CRITERIA_TXT}")

    # 1. Event metrics per ensemble.
    all_data = {}
    for ds in DATASETS:
        all_data[ds] = load_event_metrics(ds, SSI_WINDOW)
        print(f"  {ds}: {len(all_data[ds])} events")

    # 2. Shared focal-region setup (same cache as Fig10).
    setup = fig10._load_or_compute_focal_setup(all_data)
    print(f"  Focal region: {len(setup['focal_cells'])} cells")
    if len(setup['focal_cells']) == 0:
        raise RuntimeError("Empty focal region; cannot anchor events.")

    # 3. SSI calculator + FFMP seasonal rule curves (as in Fig10).
    print(f"  Fitting SSI-{SSI_WINDOW} calculator on baseline inflow...")
    ssi_calc = fit_ssi_calculator(SSI_WINDOW)
    ffmp_seasonal = fig10._build_ffmp_seasonal_lookup()
    if ffmp_seasonal is None:
        print("  Warning: FFMP boundaries unavailable; storage panels "
              "will be drawn without rule-curve lines.")

    # 4. One figure per scenario, both slots from that scenario.
    window_cache = {}
    records = []
    for ds in DATASETS:
        print(f"Scenario '{ds}' ({DATASET_LABELS[ds]}):")
        criteria = build_scenario_criteria(ds)

        # Scenario's own per-bin event counts for panel (a)'s support grid.
        _, count_grid = compute_min_storage_grid(
            all_data[ds], setup['sev_edges'], setup['mag_edges'],
            min_count=fig10.MIN_COUNT,
        )

        selected = fig10._render_version(
            SCENARIO_SUFFIX[ds], criteria, all_data, setup, ssi_calc,
            ffmp_seasonal, window_cache,
            out_dir=SI_OUTPUT_DIR, name_stem=FIG_NAME_STEM,
            support_grid=count_grid,
            support_label=f"{DATASET_LABELS_SHORT[ds]}-ensemble",
        )
        for slot in fig10.EVENT_SLOT_ORDER:
            records.append(event_record(ds, slot, selected[slot]))

    pd.DataFrame(records).to_csv(EVENTS_CSV, index=False)
    print(f"  Wrote selected-event table: {EVENTS_CSV}")
    print("Done.")


if __name__ == '__main__':
    main()
