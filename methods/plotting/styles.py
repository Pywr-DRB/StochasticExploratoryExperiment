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

# Primary color scheme for datasets
DATASET_COLORS = {
    'stationary_ensemble': '#1f77b4',           # Blue
    'climate_adjusted_low': '#d62728',          # Red (Dry scenario)
    'climate_adjusted_medium': '#9467bd',       # Purple (Medium scenario)
    'climate_adjusted_high': '#2ca02c',         # Green (Wet scenario)
}

# Alternative color scheme (if needed for specific plots)
DATASET_COLORS_ALT = {
    'stationary_ensemble': '#2E86AB',           # Blue (alternative)
    'climate_adjusted_low': '#C73E1D',          # Red-orange (Dry)
    'climate_adjusted_medium': '#A23B72',       # Magenta-purple (Medium)
    'climate_adjusted_high': '#06A77D',         # Teal (Wet)
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
