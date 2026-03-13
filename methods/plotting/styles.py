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
    'stationary_ensemble': 'Baseline Climate',
    'climate_adjusted_low': 'Mixed Future Climate',
    'climate_adjusted_medium': 'Climate Medium',
    'climate_adjusted_high': 'Wet Future Climate',
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
HISTORIC_LABEL = 'Historical'

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


# =============================================================================
# PERFORMANCE METRICS CONFIGURATION
# =============================================================================

# Annual metrics display names (for plot labels)
# These correspond to columns in the {dataset_id}_annual_metrics.csv
METRIC_DISPLAY_NAMES = {
    # Per-location shortage metrics (4 × 3 locations = 12)
    'montague_reliability': 'Montague\nReliability',
    'montague_shortage_mg': 'Montague\nShortage (MG)',
    'montague_max_consec_shortage_days': 'Max Consec.\nMontague Shortage (d)',
    'montague_max_1day_shortage_mg': 'Max 1-Day\nMontague Shortage (MG)',

    'trenton_reliability': 'Trenton\nReliability',
    'trenton_shortage_mg': 'Trenton\nShortage (MG)',
    'trenton_max_consec_shortage_days': 'Max Consec.\nTrenton Shortage (d)',
    'trenton_max_1day_shortage_mg': 'Max 1-Day\nTrenton Shortage (MG)',

    'nyc_reliability': 'NYC Diversion\nReliability',
    'nyc_shortage_mg': 'NYC Diversion\nShortage (MG)',
    'nyc_max_consec_shortage_days': 'Max Consec.\nNYC Shortage (d)',
    'nyc_max_1day_shortage_mg': 'Max 1-Day\nNYC Shortage (MG)',

    # NYC storage metrics (5)
    'nyc_min_storage_pct': 'Min NYC\nStorage (%)',
    'june1_storage_pct': 'June 1\nStorage (%)',
    'sept1_storage_pct': 'Sept 1\nStorage (%)',
    'ndays_storage_below_20pct': 'Days Storage\n< 20%',
    'ndays_storage_below_30pct': 'Days Storage\n< 30%',

    # System metrics (3)
    'nyc_contribution_mg': 'NYC\nContribution (MG)',
    'ndays_combined_stress': 'Days Combined\nStress',
    'max_zone': 'Max NYC\nDrought Zone',

    # Hashimoto simulation-level
    'hashimoto_reliability_montague': 'Hashimoto Reliability\nMontague (%)',
    'hashimoto_resiliency_montague': 'Hashimoto Resiliency\nMontague (%)',
    'hashimoto_reliability_trenton': 'Hashimoto Reliability\nTrenton (%)',
    'hashimoto_resiliency_trenton': 'Hashimoto Resiliency\nTrenton (%)',

    # Contribution analysis columns
    'annual_max_zone': 'Annual Max\nDrought Zone',
    'annual_min_storage_pct': 'Annual Min\nStorage (%)',
}

# Metric units (for determining appropriate y-axis labels)
METRIC_UNITS = {
    # Reliability (fraction 0-1)
    'montague_reliability': 'fraction',
    'trenton_reliability': 'fraction',
    'nyc_reliability': 'fraction',

    # Percentage metrics
    'nyc_min_storage_pct': 'percent',
    'june1_storage_pct': 'percent',
    'sept1_storage_pct': 'percent',
    'hashimoto_reliability_montague': 'percent',
    'hashimoto_resiliency_montague': 'percent',
    'hashimoto_reliability_trenton': 'percent',
    'hashimoto_resiliency_trenton': 'percent',
    'annual_min_storage_pct': 'percent',

    # Duration metrics (days)
    'montague_max_consec_shortage_days': 'days',
    'trenton_max_consec_shortage_days': 'days',
    'nyc_max_consec_shortage_days': 'days',
    'ndays_storage_below_20pct': 'days',
    'ndays_storage_below_30pct': 'days',
    'ndays_combined_stress': 'days',

    # Volume metrics (million gallons)
    'montague_shortage_mg': 'million_gallons',
    'montague_max_1day_shortage_mg': 'million_gallons',
    'trenton_shortage_mg': 'million_gallons',
    'trenton_max_1day_shortage_mg': 'million_gallons',
    'nyc_shortage_mg': 'million_gallons',
    'nyc_max_1day_shortage_mg': 'million_gallons',
    'nyc_contribution_mg': 'million_gallons',

    # Zone level (integer 1-6)
    'max_zone': 'zone_level',
    'annual_max_zone': 'zone_level',
}

# Y-axis labels for different metric types
Y_AXIS_LABELS = {
    'fraction': 'Reliability (0–1)',
    'percent': 'Percentage (%)',
    'days': 'Days',
    'count': 'Count',
    'million_gallons': 'Million Gallons (MG)',
    'zone_level': 'Zone Level',
    'value': 'Value',  # Generic fallback
}

# Reconstruction scaling — no longer needed for annual metrics
# (annual metrics are per water year, not aggregated counts)
RECONSTRUCTION_YEARS = 79
ENSEMBLE_YEARS = 70
RECONSTRUCTION_SCALE_FACTOR = ENSEMBLE_YEARS / RECONSTRUCTION_YEARS

# Metrics that need scaling when comparing reconstruction to ensemble
# (only applies if aggregating annual metrics to counts externally)
METRICS_TO_SCALE = []

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
