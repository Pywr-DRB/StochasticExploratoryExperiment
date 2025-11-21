"""
Centralized styling configuration for StochasticExploratoryExperiment plotting.

This module provides consistent colors, labels, markers, and other styling
parameters for all visualization scripts across the project.

Usage:
------
from methods.plotting.styles import DATASET_COLORS, DATASET_LABELS

# Use in plotting
plt.plot(data, color=DATASET_COLORS['stationary_ensemble'],
         label=DATASET_LABELS['stationary_ensemble'])
"""

# =============================================================================
# DATASET COLORS
# =============================================================================

# Primary color scheme for datasets - Colorblind-friendly palette
# Based on Wong (2011) "Points of view: Color blindness" Nature Methods
# These colors are distinguishable for deuteranopia, protanopia, and tritanopia
DATASET_COLORS = {
    'stationary_ensemble': '#0072B2',           # Blue (baseline/reference)
    'climate_adjusted_low': '#D55E00',          # Vermilion/Orange (Dry scenario)
    'climate_adjusted_medium': '#CC79A7',       # Reddish purple (Medium scenario)
    'climate_adjusted_high': '#009E73',         # Bluish green (Wet scenario)
}

# Alternative color scheme - also colorblind-friendly
# Using IBM Design Language accessible palette
DATASET_COLORS_ALT = {
    'stationary_ensemble': '#648FFF',           # Ultramarine blue (baseline)
    'climate_adjusted_low': '#FE6100',          # Orange (Dry)
    'climate_adjusted_medium': '#DC267F',       # Magenta (Medium)
    'climate_adjusted_high': '#785EF0',         # Purple (Wet)
}

# Historic/observed data color
HISTORIC_COLOR = '#000000'  # Black

# =============================================================================
# DATASET LABELS
# =============================================================================

# Standard labels for datasets
DATASET_LABELS = {
    'stationary_ensemble': 'Stationary',
    'climate_adjusted_low': 'Climate Low',
    'climate_adjusted_medium': 'Climate Medium',
    'climate_adjusted_high': 'Climate High',
}

# Short labels (for tight layouts)
DATASET_LABELS_SHORT = {
    'stationary_ensemble': 'Stationary',
    'climate_adjusted_low': 'Low',
    'climate_adjusted_medium': 'Medium',
    'climate_adjusted_high': 'High',
}

# Descriptive labels (for titles/captions)
DATASET_LABELS_DESCRIPTIVE = {
    'stationary_ensemble': 'Stationary Ensemble (Historical Statistics)',
    'climate_adjusted_low': 'Climate Adjusted - Low (Driest Scenario)',
    'climate_adjusted_medium': 'Climate Adjusted - Medium (Mid-range Scenario)',
    'climate_adjusted_high': 'Climate Adjusted - High (Wettest Scenario)',
}

# Historic/observed label
HISTORIC_LABEL = 'Historic'

# =============================================================================
# DATASET ORDER
# =============================================================================

# Standard order for displaying datasets (for consistent ordering in legends, panels, etc.)
DATASET_ORDER = [
    'stationary_ensemble',
    'climate_adjusted_low',
    'climate_adjusted_medium',
    'climate_adjusted_high',
]

# =============================================================================
# PLOT MARKERS AND LINE STYLES
# =============================================================================

# Markers for datasets
DATASET_MARKERS = {
    'stationary_ensemble': 'o',
    'climate_adjusted_low': 's',
    'climate_adjusted_medium': '^',
    'climate_adjusted_high': 'D',
}

# Line styles for datasets
DATASET_LINESTYLES = {
    'stationary_ensemble': '-',
    'climate_adjusted_low': '--',
    'climate_adjusted_medium': '-.',
    'climate_adjusted_high': ':',
}

# Historic marker and line style
HISTORIC_MARKER = 'o'
HISTORIC_LINESTYLE = '-'

# =============================================================================
# ALPHA VALUES (TRANSPARENCY)
# =============================================================================

# Standard alpha values for consistency
ALPHA_FILL = 0.3        # For fill_between areas
ALPHA_LINE = 0.8        # For line plots
ALPHA_SCATTER = 0.7     # For scatter plots
ALPHA_BAR = 0.8         # For bar plots

# =============================================================================
# LINE WIDTHS
# =============================================================================

LINEWIDTH_THIN = 1.0
LINEWIDTH_MEDIUM = 2.0
LINEWIDTH_THICK = 2.5

# =============================================================================
# COLORMAPS
# =============================================================================

# Standard colormaps for different plot types
CMAP_SEQUENTIAL = 'viridis'      # For sequential data
CMAP_DIVERGING = 'BrBG'          # For diverging data (differences)
CMAP_HEATMAP = 'magma'           # For heatmaps/return periods

# =============================================================================
# FIGURE SIZES
# =============================================================================

# Standard figure sizes (width, height) in inches
FIGSIZE_SINGLE = (8, 6)          # Single panel
FIGSIZE_DOUBLE = (14, 6)         # Two panels side-by-side
FIGSIZE_TRIPLE = (18, 6)         # Three panels side-by-side
FIGSIZE_QUAD = (14, 10)          # Four panels (2x2 grid)
FIGSIZE_LARGE = (16, 10)         # Large multi-panel figure

# =============================================================================
# FONT SIZES
# =============================================================================

FONTSIZE_SMALL = 9
FONTSIZE_MEDIUM = 10
FONTSIZE_LARGE = 11
FONTSIZE_TITLE = 14
FONTSIZE_SUPTITLE = 16
FONTSIZE_LABEL = 12
FONTSIZE_LEGEND = 10

# =============================================================================
# DPI SETTINGS
# =============================================================================

DPI_SCREEN = 100        # For screen display
DPI_PRINT = 300         # For standard print quality
DPI_HIGH = 400          # For high-quality print/publication

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_dataset_color(dataset_id, alternative=False):
    """
    Get color for a dataset.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    alternative : bool, optional
        If True, use alternative color scheme (default: False)

    Returns
    -------
    str
        Color hex code
    """
    colors = DATASET_COLORS_ALT if alternative else DATASET_COLORS
    return colors.get(dataset_id, '#808080')  # Default to gray if not found


def get_dataset_label(dataset_id, style='standard'):
    """
    Get label for a dataset.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    style : str, optional
        Label style: 'standard', 'short', or 'descriptive' (default: 'standard')

    Returns
    -------
    str
        Dataset label
    """
    if style == 'short':
        labels = DATASET_LABELS_SHORT
    elif style == 'descriptive':
        labels = DATASET_LABELS_DESCRIPTIVE
    else:
        labels = DATASET_LABELS

    return labels.get(dataset_id, dataset_id)


def get_all_dataset_colors(alternative=False):
    """
    Get colors for all datasets in standard order.

    Parameters
    ----------
    alternative : bool, optional
        If True, use alternative color scheme (default: False)

    Returns
    -------
    list
        List of colors in standard dataset order
    """
    return [get_dataset_color(did, alternative) for did in DATASET_ORDER]


def get_all_dataset_labels(style='standard'):
    """
    Get labels for all datasets in standard order.

    Parameters
    ----------
    style : str, optional
        Label style: 'standard', 'short', or 'descriptive' (default: 'standard')

    Returns
    -------
    list
        List of labels in standard dataset order
    """
    return [get_dataset_label(did, style) for did in DATASET_ORDER]


def apply_publication_style():
    """
    Apply publication-quality style settings to matplotlib.

    This function configures matplotlib with publication-ready defaults.
    Call this at the beginning of plotting scripts for consistent styling.

    Example:
    --------
    >>> from methods.plotting.styles import apply_publication_style
    >>> apply_publication_style()
    >>> fig, ax = plt.subplots()
    >>> # ... your plotting code
    """
    import matplotlib.pyplot as plt

    # Set style parameters
    plt.rcParams.update({
        'font.size': FONTSIZE_MEDIUM,
        'axes.labelsize': FONTSIZE_LABEL,
        'axes.titlesize': FONTSIZE_TITLE,
        'xtick.labelsize': FONTSIZE_MEDIUM,
        'ytick.labelsize': FONTSIZE_MEDIUM,
        'legend.fontsize': FONTSIZE_LEGEND,
        'figure.titlesize': FONTSIZE_SUPTITLE,
        'figure.dpi': DPI_SCREEN,
        'savefig.dpi': DPI_PRINT,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'axes.axisbelow': True,
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.0,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'legend.frameon': True,
        'legend.fancybox': True,
        'legend.shadow': True,
        'figure.constrained_layout.use': False,
    })


def create_dataset_legend_handles():
    """
    Create legend handles for all datasets with consistent styling.

    Returns
    -------
    list of matplotlib.lines.Line2D
        Legend handles for all datasets in standard order

    Example:
    --------
    >>> from methods.plotting.styles import create_dataset_legend_handles
    >>> handles = create_dataset_legend_handles()
    >>> labels = get_all_dataset_labels()
    >>> ax.legend(handles, labels, loc='best')
    """
    from matplotlib.lines import Line2D

    handles = []
    for dataset_id in DATASET_ORDER:
        handle = Line2D([0], [0],
                       color=DATASET_COLORS[dataset_id],
                       marker=DATASET_MARKERS[dataset_id],
                       linestyle=DATASET_LINESTYLES[dataset_id],
                       linewidth=LINEWIDTH_MEDIUM,
                       markersize=8,
                       label=DATASET_LABELS[dataset_id])
        handles.append(handle)

    return handles


# =============================================================================
# SCENARIO-SPECIFIC STYLING
# =============================================================================

def get_scenario_style(dataset_id, include_historic=False):
    """
    Get a complete style dictionary for a dataset.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    include_historic : bool, optional
        If True and dataset_id is 'historic', return historic styling

    Returns
    -------
    dict
        Style dictionary with keys: color, label, marker, linestyle, alpha

    Example:
    --------
    >>> style = get_scenario_style('stationary_ensemble')
    >>> ax.plot(x, y, **style)  # Unpack style dict directly
    """
    if dataset_id == 'historic' or dataset_id == 'observed':
        return {
            'color': HISTORIC_COLOR,
            'label': HISTORIC_LABEL,
            'marker': HISTORIC_MARKER,
            'linestyle': HISTORIC_LINESTYLE,
            'linewidth': LINEWIDTH_THICK,
            'alpha': 1.0,
            'zorder': 10,  # Draw on top
        }

    return {
        'color': DATASET_COLORS.get(dataset_id, '#808080'),
        'label': DATASET_LABELS.get(dataset_id, dataset_id),
        'marker': DATASET_MARKERS.get(dataset_id, 'o'),
        'linestyle': DATASET_LINESTYLES.get(dataset_id, '-'),
        'linewidth': LINEWIDTH_MEDIUM,
        'alpha': ALPHA_LINE,
        'zorder': 5,
    }


# =============================================================================
# PERFORMANCE METRICS CONFIGURATION
# =============================================================================

# Metric display names (for plot labels)
METRIC_DISPLAY_NAMES = {
    # Flow Reliability - Montague
    'years_reliable_montague': 'Years Montague\nReliable (>90%)',
    'years_reliable_montague_95': 'Years Montague\nReliable (>95%)',
    'mean_annual_montague_reliability': 'Mean Annual\nMontague Reliability',
    'min_annual_montague_reliability': 'Worst Annual\nMontague Reliability',
    'total_montague_shortage_mg': 'Total Montague\nShortage (MG)',
    'mean_annual_montague_shortage_mg': 'Mean Annual\nMontague Shortage',

    # Flow Reliability - Trenton
    'years_reliable_trenton': 'Years Trenton\nReliable (>90%)',
    'years_reliable_trenton_95': 'Years Trenton\nReliable (>95%)',
    'mean_annual_trenton_reliability': 'Mean Annual\nTrenton Reliability',
    'total_trenton_shortage_mg': 'Total Trenton\nShortage (MG)',
    'mean_annual_trenton_shortage_mg': 'Mean Annual\nTrenton Shortage',

    # NYC Storage - Thresholds
    'years_above_30pct': 'Years Min\nStorage >30%',
    'years_above_20pct': 'Years Min\nStorage >20%',
    'years_above_10pct': 'Years Min\nStorage >10%',
    'years_below_10pct': 'Years Min\nStorage ≤10%',

    # NYC Storage - Key Dates
    'years_high_storage_june1': 'Years June 1\nStorage ≥95%',
    'years_high_storage_june1_90': 'Years June 1\nStorage ≥90%',
    'mean_june1_storage_pct': 'Mean June 1\nStorage (%)',
    'mean_sept1_storage_pct': 'Mean Sept 1\nStorage (%)',
    'years_low_carryover': 'Years Sept 1\nStorage <50%',
    'years_low_carryover_40': 'Years Sept 1\nStorage <40%',

    # NYC Storage - Statistics
    'mean_storage_pct': 'Mean Storage (%)',
    'median_storage_pct': 'Median Storage (%)',
    'min_storage_pct': 'Min Storage (%)',
    'max_storage_pct': 'Max Storage (%)',
    'std_storage_pct': 'Storage Std Dev (%)',
    'pct_days_storage_below_30': '% Days\nStorage <30%',
    'pct_days_storage_below_20': '% Days\nStorage <20%',
    'mean_annual_storage_range': 'Mean Annual\nStorage Range (%)',

    # Water Supply Reliability
    'pct_days_nyc_diversion_shortage': '% Days NYC\nDiversion Shortage',
    'total_nyc_diversion_shortage_mg': 'Total NYC\nDiversion Shortage',
    'mean_annual_nyc_diversion_shortage_mg': 'Mean Annual NYC\nDiversion Shortage',
    'max_daily_nyc_diversion_shortage_mg': 'Max Daily NYC\nDiversion Shortage',
    'years_no_nyc_shortage': 'Years No\nNYC Shortage',
    'years_minor_nyc_shortage': 'Years Minor\nNYC Shortage',

    # Drought Characteristics
    'max_consecutive_drought_days': 'Max Consecutive\nDrought (days)',
    'mean_drought_duration_days': 'Mean Drought\nDuration (days)',
    'n_drought_events': 'Number of\nDrought Events',
    'n_major_droughts': 'Number of\nMajor Droughts',
    'n_severe_droughts': 'Number of\nSevere Droughts',
    'worst_drought_max_daily_shortage_mg': 'Worst Drought\nPeak Shortage',
    'max_consecutive_drought_days_trenton': 'Max Consecutive\nTrenton Drought',
    'n_drought_events_trenton': 'Number of\nTrenton Droughts',
    'pct_days_combined_stress': '% Days Combined\nSystem Stress',

    # NYC Contributions
    'mean_annual_nyc_contribution_mg': 'Mean Annual NYC\nContribution (MG)',
    'max_annual_nyc_contribution_mg': 'Max Annual NYC\nContribution (MG)',
    'min_annual_nyc_contribution_mg': 'Min Annual NYC\nContribution (MG)',
    'std_annual_nyc_contribution_mg': 'NYC Contribution\nStd Dev (MG)',
    'total_nyc_contribution_mg': 'Total NYC\nContribution (MG)',
    'pct_days_nyc_contribution': '% Days NYC\nContribution',
    'n_days_high_nyc_contribution': 'Days High NYC\nContribution (>100 MGD)',

    # System Balance
    'nyc_contribution_to_shortage_ratio': 'NYC Contribution /\nShortage Ratio',
    'years_high_storage_and_reliable': 'Years High Storage\n& Reliable',
    'years_vulnerable': 'Years\nVulnerable',

    # Legacy
    'years_reliable': 'Years Montague\nReliable',
    'years_high_storage': 'Years June 1\nStorage High',
}

# Metric units (for determining appropriate y-axis labels)
METRIC_UNITS = {
    # Year count metrics
    'years_reliable': 'years',
    'years_reliable_montague': 'years',
    'years_reliable_montague_95': 'years',
    'years_reliable_trenton': 'years',
    'years_reliable_trenton_95': 'years',
    'years_high_storage': 'years',
    'years_high_storage_june1': 'years',
    'years_high_storage_june1_90': 'years',
    'years_above_30pct': 'years',
    'years_above_20pct': 'years',
    'years_above_10pct': 'years',
    'years_below_10pct': 'years',
    'years_low_carryover': 'years',
    'years_low_carryover_40': 'years',
    'years_no_nyc_shortage': 'years',
    'years_minor_nyc_shortage': 'years',
    'years_high_storage_and_reliable': 'years',
    'years_vulnerable': 'years',

    # Percentage metrics
    'mean_sept1_storage_pct': 'percent',
    'mean_june1_storage_pct': 'percent',
    'mean_storage_pct': 'percent',
    'median_storage_pct': 'percent',
    'min_storage_pct': 'percent',
    'max_storage_pct': 'percent',
    'std_storage_pct': 'percent',
    'pct_days_storage_below_30': 'percent',
    'pct_days_storage_below_20': 'percent',
    'pct_days_nyc_diversion_shortage': 'percent',
    'pct_days_nyc_contribution': 'percent',
    'pct_days_combined_stress': 'percent',
    'mean_annual_storage_range': 'percent',
    'mean_annual_montague_reliability': 'percent',
    'min_annual_montague_reliability': 'percent',
    'mean_annual_trenton_reliability': 'percent',

    # Duration metrics (days)
    'max_consecutive_drought_days': 'days',
    'max_consecutive_drought_days_trenton': 'days',
    'mean_drought_duration_days': 'days',
    'n_days_high_nyc_contribution': 'days',

    # Count metrics
    'n_drought_events': 'count',
    'n_drought_events_trenton': 'count',
    'n_major_droughts': 'count',
    'n_severe_droughts': 'count',

    # Volume metrics (million gallons)
    'mean_annual_nyc_contribution_mg': 'million_gallons',
    'max_annual_nyc_contribution_mg': 'million_gallons',
    'min_annual_nyc_contribution_mg': 'million_gallons',
    'std_annual_nyc_contribution_mg': 'million_gallons',
    'total_nyc_contribution_mg': 'million_gallons',
    'total_montague_shortage_mg': 'million_gallons',
    'mean_annual_montague_shortage_mg': 'million_gallons',
    'total_trenton_shortage_mg': 'million_gallons',
    'mean_annual_trenton_shortage_mg': 'million_gallons',
    'total_nyc_diversion_shortage_mg': 'million_gallons',
    'mean_annual_nyc_diversion_shortage_mg': 'million_gallons',
    'max_daily_nyc_diversion_shortage_mg': 'million_gallons',
    'worst_drought_max_daily_shortage_mg': 'million_gallons',

    # Ratio metrics
    'nyc_contribution_to_shortage_ratio': 'ratio',
}

# Y-axis labels for different metric types
Y_AXIS_LABELS = {
    'years': 'Number of Years (out of 70)',
    'percent': 'Percentage (%)',
    'days': 'Days',
    'count': 'Count',
    'million_gallons': 'Million Gallons (MG)',
    'ratio': 'Ratio',
    'value': 'Value',  # Generic fallback
}

# Reconstruction scaling
RECONSTRUCTION_YEARS = 79
ENSEMBLE_YEARS = 70
RECONSTRUCTION_SCALE_FACTOR = ENSEMBLE_YEARS / RECONSTRUCTION_YEARS  # 70/79 ≈ 0.886

# Metrics that should be scaled when comparing reconstruction to ensemble
# (These are year-count metrics that need adjustment for different time periods)
METRICS_TO_SCALE = [
    'years_reliable',
    'years_reliable_montague',
    'years_reliable_montague_95',
    'years_reliable_trenton',
    'years_reliable_trenton_95',
    'years_high_storage',
    'years_high_storage_june1',
    'years_high_storage_june1_90',
    'years_above_30pct',
    'years_above_20pct',
    'years_above_10pct',
    'years_below_10pct',
    'years_low_carryover',
    'years_low_carryover_40',
    'years_no_nyc_shortage',
    'years_minor_nyc_shortage',
    'years_high_storage_and_reliable',
    'years_vulnerable',
]

# Historic reconstruction marker style
HISTORIC_MARKER_STYLE = {
    'marker': 'D',
    'color': 'red',
    's': 100,
    'edgecolors': 'darkred',
    'linewidths': 2,
    'zorder': 10,
}


def get_ylabel_for_metrics(metric_list):
    """
    Determine appropriate y-axis label for a list of metrics.
    If all metrics have same units, return that unit's label.
    Otherwise, return a generic label.

    Parameters
    ----------
    metric_list : list
        List of metric names

    Returns
    -------
    str
        Appropriate y-axis label
    """
    units = set(METRIC_UNITS.get(m, 'value') for m in metric_list)

    if len(units) == 1:
        unit = units.pop()
        return Y_AXIS_LABELS.get(unit, 'Value')
    else:
        return 'Value'
