"""
Sankey-Parallel Coordinate hybrid figure renderer.

Renders a figure with multiple horizontal axes (parallel coordinates)
connected by Sankey-style flow bands colored by classification.

Each axis represents a metric, binned into discrete ranges.
Bin widths are proportional to sample count (not value).
Flows between adjacent axes show how samples redistribute,
colored by satisficing classification.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from methods.plotting.styles import (
    FONTSIZE_SMALL, FONTSIZE_MEDIUM, FONTSIZE_LABEL, FONTSIZE_TITLE,
    DPI_HIGH, apply_publication_style,
)


# =============================================================================
# CLASSIFICATION COLORS (consistent with F11 satisficing categories)
# =============================================================================

CLASSIFICATION_COLORS = {
    'all_pass': '#888888',         # Grey - all criteria met
    'storage_fail': '#ff7f0e',     # Orange - storage failed only
    'montague_fail': '#d62728',    # Red - Montague failed only
    'both_fail': '#7f0000',        # Dark red - both failed
}

CLASSIFICATION_LABELS = {
    'all_pass': 'Satisficing',
    'storage_fail': 'Storage Failure',
    'montague_fail': 'Montague Failure',
    'both_fail': 'Both Fail',
}

# Drawing order: pass in back, failures on top
CLASSIFICATION_DRAW_ORDER = ['all_pass', 'storage_fail', 'montague_fail', 'both_fail']


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class AxisConfig:
    """Configuration for a single horizontal axis in the figure."""
    metric: str             # Column name in event_metrics CSV
    label: str              # Display label (right side of axis)
    bin_edges: list         # Bin edge values, or 'quantile' for auto-tertiles
    bin_labels: list = None # Optional custom labels per bin (len = len(bin_edges)-1)

    def resolve_edges(self, series):
        """
        Resolve bin edges from data if 'quantile' was specified.

        Parameters
        ----------
        series : pd.Series
            Data values for this metric

        Returns
        -------
        list
            Resolved numeric bin edges
        """
        if self.bin_edges == 'quantile':
            # use every 10th percentile to get 10 bins
            q = series.quantile(np.linspace(0, 1, 11)).values
            # Ensure unique edges by adding small epsilon
            edges = list(np.unique(q))
            if len(edges) < 3:
                edges = [series.min(), series.median(), series.max()]
            # Extend edges slightly to capture all data
            edges[0] = edges[0] - 1e-6
            edges[-1] = edges[-1] + 1e-6
            return edges
        else:
            return list(self.bin_edges)


@dataclass
class SankeyFigureConfig:
    """Configuration for the full Sankey-Parallel Coordinate figure."""
    axes: List[AxisConfig]
    classification_col: str = 'classification'
    classification_colors: Dict[str, str] = None
    classification_labels: Dict[str, str] = None
    min_bin_width_frac: float = 0.05   # Minimum bin width as fraction of total
    flow_alpha: float = 0.50
    bin_alpha: float = 0.85
    figsize: Tuple = (14, 10)
    bin_height: float = 0.06           # Height of bin rectangles in axes coords
    axis_gap: float = 0.12             # Vertical gap between axes
    bin_gap: float = 0.01              # Horizontal gap between bins
    edge_pad: float = 0.02             # Padding on left/right edges
    bin_color: str = '#ffffff'         # White bin rectangle fill
    bin_edge_color: str = '#666666'    # Bin rectangle edge color

    def __post_init__(self):
        if self.classification_colors is None:
            self.classification_colors = CLASSIFICATION_COLORS.copy()
        if self.classification_labels is None:
            self.classification_labels = CLASSIFICATION_LABELS.copy()


# =============================================================================
# BINNING AND FLOW COMPUTATION
# =============================================================================

def bin_samples(metrics_df, axis_configs):
    """
    Assign each sample to a bin on each axis.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Event metrics DataFrame
    axis_configs : list of AxisConfig
        Ordered axis configurations

    Returns
    -------
    bin_assignments : pd.DataFrame
        Columns: axis_0, axis_1, ..., axis_N (integer bin indices, 0-based)
    resolved_edges : list of list
        Resolved bin edges for each axis
    """
    bin_assignments = pd.DataFrame(index=metrics_df.index)
    resolved_edges = []

    for i, config in enumerate(axis_configs):
        series = metrics_df[config.metric]
        edges = config.resolve_edges(series)
        resolved_edges.append(edges)

        # pd.cut returns bin indices; use labels=False for integer bins
        bins = pd.cut(series, bins=edges, labels=False, include_lowest=True)
        # Handle values outside bin range (assign to nearest bin)
        bins = bins.fillna(method='ffill').fillna(method='bfill')
        # If still NaN (all outside), assign to bin 0
        bins = bins.fillna(0).astype(int)
        bin_assignments[f'axis_{i}'] = bins

    return bin_assignments, resolved_edges


def compute_flows(bin_assignments, classifications, axis_configs):
    """
    Compute flow counts between adjacent axes, grouped by classification.

    Parameters
    ----------
    bin_assignments : pd.DataFrame
        Columns axis_0, axis_1, ... with integer bin indices
    classifications : pd.Series
        Classification label per sample
    axis_configs : list of AxisConfig

    Returns
    -------
    flows : dict
        flows[(axis_i, src_bin, tgt_bin, classification)] = count
    """
    flows = {}
    n_axes = len(axis_configs)

    for i in range(n_axes - 1):
        src_col = f'axis_{i}'
        tgt_col = f'axis_{i+1}'
        grouped = pd.DataFrame({
            'src': bin_assignments[src_col],
            'tgt': bin_assignments[tgt_col],
            'cls': classifications,
        }).groupby(['src', 'tgt', 'cls']).size()

        for (src, tgt, cls), count in grouped.items():
            flows[(i, int(src), int(tgt), cls)] = count

    return flows


def compute_bin_counts(bin_assignments, classifications, axis_configs):
    """
    Compute sample counts per bin per axis, and per-classification breakdown.

    Returns
    -------
    bin_counts : dict
        bin_counts[(axis_i, bin_j)] = total count
    bin_class_counts : dict
        bin_class_counts[(axis_i, bin_j, classification)] = count
    """
    bin_counts = {}
    bin_class_counts = {}
    n_axes = len(axis_configs)

    for i in range(n_axes):
        col = f'axis_{i}'
        counts = bin_assignments[col].value_counts().to_dict()
        for b, c in counts.items():
            bin_counts[(i, int(b))] = c

        # Per-classification
        grouped = pd.DataFrame({
            'bin': bin_assignments[col],
            'cls': classifications,
        }).groupby(['bin', 'cls']).size()
        for (b, cls), c in grouped.items():
            bin_class_counts[(i, int(b), cls)] = c

    return bin_counts, bin_class_counts


def compute_bin_widths(bin_counts, n_bins_per_axis, config):
    """
    Compute normalized bin widths proportional to count, with minimum floor.

    Parameters
    ----------
    bin_counts : dict
        bin_counts[(axis_i, bin_j)] = count
    n_bins_per_axis : list of int
        Number of bins per axis
    config : SankeyFigureConfig

    Returns
    -------
    bin_widths : dict
        bin_widths[(axis_i, bin_j)] = normalized width (0 to 1)
    bin_x_starts : dict
        bin_x_starts[(axis_i, bin_j)] = x start position (0 to 1)
    """
    total_width = 1.0 - 2 * config.edge_pad
    bin_widths = {}
    bin_x_starts = {}

    for ax_i, n_bins in enumerate(n_bins_per_axis):
        # Get counts for this axis
        counts = []
        for b in range(n_bins):
            counts.append(bin_counts.get((ax_i, b), 0))
        total = sum(counts)
        if total == 0:
            total = 1

        # Available width after gaps
        gap_space = config.bin_gap * (n_bins - 1)
        available = total_width - gap_space

        # Raw proportional widths
        raw = [c / total * available for c in counts]

        # Apply minimum width floor
        min_w = config.min_bin_width_frac * available
        adjusted = [max(r, min_w) for r in raw]

        # Renormalize to fit in available space
        adj_total = sum(adjusted)
        if adj_total > 0:
            scale = available / adj_total
            adjusted = [a * scale for a in adjusted]

        # Compute x start positions
        x = config.edge_pad
        for b in range(n_bins):
            bin_widths[(ax_i, b)] = adjusted[b]
            bin_x_starts[(ax_i, b)] = x
            x += adjusted[b] + config.bin_gap

    return bin_widths, bin_x_starts


# =============================================================================
# RENDERING
# =============================================================================

def _draw_flow_band(ax, x0_l, x0_r, y0, x1_l, x1_r, y1, color, alpha):
    """
    Draw a curved Sankey-style band between two segments using Bezier curves.

    (x0_l, x0_r) define the source segment at y0 (bottom of source bin).
    (x1_l, x1_r) define the target segment at y1 (top of target bin).
    """
    mid_y = (y0 + y1) / 2.0

    verts = [
        (x0_l, y0),       # Start: top-left of source
        (x0_l, mid_y),    # Control point 1
        (x1_l, mid_y),    # Control point 2
        (x1_l, y1),       # End: top-left of target
        (x1_r, y1),       # Bottom-right of target
        (x1_r, mid_y),    # Control point 3
        (x0_r, mid_y),    # Control point 4
        (x0_r, y0),       # Bottom-right of source
        (x0_l, y0),       # Close path
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CLOSEPOLY,
    ]
    path = Path(verts, codes)
    patch = PathPatch(path, facecolor=color, edgecolor='none',
                      alpha=alpha, zorder=2)
    ax.add_patch(patch)


def plot_sankey_parallel(metrics_df, config, output_path=None, show=False,
                          title=None):
    """
    Generate the Sankey-Parallel Coordinate figure.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Event metrics with columns matching axis_config metrics and
        a classification column.
    config : SankeyFigureConfig
        Full figure configuration.
    output_path : str, optional
        If provided, save figure to this path.
    show : bool
        If True, call plt.show().
    title : str, optional
        Figure title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    apply_publication_style()

    axis_configs = config.axes
    n_axes = len(axis_configs)
    classifications = metrics_df[config.classification_col]

    # --- Step 1: Bin samples ---
    bin_assignments, resolved_edges = bin_samples(metrics_df, axis_configs)
    n_bins_per_axis = [len(edges) - 1 for edges in resolved_edges]

    # --- Step 2: Compute counts and flows ---
    bin_counts, bin_class_counts = compute_bin_counts(
        bin_assignments, classifications, axis_configs
    )
    flows = compute_flows(bin_assignments, classifications, axis_configs)
    bin_widths, bin_x_starts = compute_bin_widths(
        bin_counts, n_bins_per_axis, config
    )

    # --- Step 3: Compute y positions for each axis ---
    # Axes are laid out top-to-bottom. axis 0 is at the top.
    # Auto-scale so all axes fit within [edge_pad, 1 - edge_pad]
    top_margin = config.edge_pad + 0.03   # Extra room for title
    bot_margin = config.edge_pad + 0.04   # Extra room for legend below
    usable = 1.0 - top_margin - bot_margin
    # Total content: n_axes bins + (n_axes-1) gaps
    raw_height = n_axes * config.bin_height + (n_axes - 1) * config.axis_gap
    if raw_height > usable:
        scale = usable / raw_height
        eff_bin_h = config.bin_height * scale
        eff_gap = config.axis_gap * scale
    else:
        eff_bin_h = config.bin_height
        eff_gap = config.axis_gap

    y_positions = {}
    for i in range(n_axes):
        y_positions[i] = (1.0 - top_margin) - i * (eff_bin_h + eff_gap) - eff_bin_h / 2

    # --- Step 4: Create figure ---
    fig, ax = plt.subplots(1, 1, figsize=config.figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('auto')
    ax.axis('off')

    if title:
        ax.set_title(title, fontsize=FONTSIZE_TITLE, fontweight='bold', pad=15)

    # --- Step 5: Draw flows between adjacent axes ---
    # Track cursors for stacking flows within each bin
    # Source bins: track bottom cursor (flows leave from bottom of bin)
    # Target bins: track top cursor (flows arrive at top of bin)
    for ax_i in range(n_axes - 1):
        src_y_bottom = y_positions[ax_i] - eff_bin_h / 2
        tgt_y_top = y_positions[ax_i + 1] + eff_bin_h / 2

        # Initialize cursors for each bin
        src_cursors = {}  # (ax_i, bin) -> current x offset within bin
        tgt_cursors = {}  # (ax_i+1, bin) -> current x offset within bin
        for b in range(n_bins_per_axis[ax_i]):
            src_cursors[b] = bin_x_starts.get((ax_i, b), 0)
        for b in range(n_bins_per_axis[ax_i + 1]):
            tgt_cursors[b] = bin_x_starts.get((ax_i + 1, b), 0)

        # Sort flows: draw 'all_pass' first (background), then failures on top
        flow_keys = []
        for (fi, src, tgt, cls), count in flows.items():
            if fi == ax_i:
                flow_keys.append((fi, src, tgt, cls, count))

        # Sort by draw order
        cls_order = {c: i for i, c in enumerate(CLASSIFICATION_DRAW_ORDER)}
        flow_keys.sort(key=lambda x: cls_order.get(x[3], 99))

        for (fi, src, tgt, cls, count) in flow_keys:
            if count == 0:
                continue

            # Source segment width proportional to flow count relative to source bin
            src_total = bin_counts.get((ax_i, src), 1)
            tgt_total = bin_counts.get((ax_i + 1, tgt), 1)
            src_bin_w = bin_widths.get((ax_i, src), 0)
            tgt_bin_w = bin_widths.get((ax_i + 1, tgt), 0)

            flow_src_w = (count / src_total) * src_bin_w if src_total > 0 else 0
            flow_tgt_w = (count / tgt_total) * tgt_bin_w if tgt_total > 0 else 0

            x0_l = src_cursors[src]
            x0_r = x0_l + flow_src_w
            x1_l = tgt_cursors[tgt]
            x1_r = x1_l + flow_tgt_w

            color = config.classification_colors.get(cls, '#cccccc')
            _draw_flow_band(ax, x0_l, x0_r, src_y_bottom,
                           x1_l, x1_r, tgt_y_top,
                           color, config.flow_alpha)

            # Advance cursors
            src_cursors[src] = x0_r
            tgt_cursors[tgt] = x1_r

    # --- Step 6: Draw bin rectangles ---
    for ax_i in range(n_axes):
        y_center = y_positions[ax_i]
        y_bot = y_center - eff_bin_h / 2
        edges = resolved_edges[ax_i]
        ac = axis_configs[ax_i]

        for b in range(n_bins_per_axis[ax_i]):
            x_start = bin_x_starts.get((ax_i, b), 0)
            w = bin_widths.get((ax_i, b), 0)
            count = bin_counts.get((ax_i, b), 0)

            # Draw bin rectangle
            rect = mpatches.FancyBboxPatch(
                (x_start, y_bot), w, eff_bin_h,
                boxstyle="round,pad=0.003",
                facecolor=config.bin_color,
                edgecolor=config.bin_edge_color,
                linewidth=0.8,
                alpha=config.bin_alpha,
                zorder=3,
            )
            ax.add_patch(rect)

            # Bin label
            if ac.bin_labels and b < len(ac.bin_labels):
                label_text = ac.bin_labels[b]
            else:
                lo = edges[b]
                hi = edges[b + 1]
                label_text = f"{lo:.1f}-{hi:.1f}"

            # Count annotation below label
            display_text = f"{label_text}\n(n={count})"
            ax.text(x_start + w / 2, y_center, display_text,
                    ha='center', va='center',
                    fontsize=FONTSIZE_SMALL - 1,
                    fontweight='normal',
                    zorder=4)

        # Axis label on right
        ax.text(1.0 - config.edge_pad / 2 + 0.02, y_center, ac.label,
                ha='left', va='center',
                fontsize=FONTSIZE_MEDIUM,
                fontweight='bold',
                zorder=5)

    # --- Step 7: Legend ---
    legend_handles = []
    for cls in CLASSIFICATION_DRAW_ORDER:
        if cls in classifications.values:
            color = config.classification_colors.get(cls, '#cccccc')
            label = config.classification_labels.get(cls, cls)
            patch = mpatches.Patch(facecolor=color, edgecolor='none',
                                   alpha=config.flow_alpha + 0.15, label=label)
            legend_handles.append(patch)

    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper center',
                  fontsize=FONTSIZE_MEDIUM, frameon=True,
                  fancybox=True, framealpha=0.9,
                  ncol=len(legend_handles),
                  bbox_to_anchor=(0.5, -0.02))

    # --- Step 8: Total sample count annotation ---
    n_total = len(metrics_df)
    ax.text(config.edge_pad, 1.0 - config.edge_pad / 2,
            f"N = {n_total} drought events",
            ha='left', va='top',
            fontsize=FONTSIZE_SMALL,
            fontstyle='italic',
            zorder=5)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=DPI_HIGH, bbox_inches='tight',
                    facecolor='white')
        print(f"  Saved figure: {output_path}")

    if show:
        plt.show()

    return fig
