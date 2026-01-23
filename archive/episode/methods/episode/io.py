"""
Input/output functions for episode analysis data.

This module handles saving and loading episode analysis outputs
including weekly time series, episodes, links, and climatology.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from .config import EpisodeAnalysisConfig


def save_episode_outputs(
    weekly_ts: pd.DataFrame,
    episodes: pd.DataFrame,
    episode_links: pd.DataFrame,
    climatology: pd.DataFrame,
    config: 'EpisodeAnalysisConfig'
) -> None:
    """
    Save all episode analysis outputs to files.

    Parameters
    ----------
    weekly_ts : pd.DataFrame
        Weekly time series with derived variables
    episodes : pd.DataFrame
        All episodes with features and classification
    episode_links : pd.DataFrame
        Episode progression relationships
    climatology : pd.DataFrame
        Weekly climatology statistics
    config : EpisodeAnalysisConfig
        Configuration object with output_dir
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_id = config.dataset_id

    # Save weekly time series
    weekly_path = output_dir / f"{dataset_id}_weekly_timeseries.parquet"
    weekly_ts.to_parquet(weekly_path, index=False)
    print(f"  Saved: {weekly_path}")

    # Save all episodes
    episodes_path = output_dir / f"{dataset_id}_episodes_all.parquet"
    episodes.to_parquet(episodes_path, index=False)
    print(f"  Saved: {episodes_path}")

    # Save stress episodes only (primary analysis set)
    stress_types = ['E1', 'E1d', 'E1c']
    stress_episodes = episodes[episodes['episode_type'].isin(stress_types)]
    stress_path = output_dir / f"{dataset_id}_episodes_stress.parquet"
    stress_episodes.to_parquet(stress_path, index=False)
    print(f"  Saved: {stress_path}")

    # Save episode links
    if len(episode_links) > 0:
        links_path = output_dir / f"{dataset_id}_episode_links.parquet"
        episode_links.to_parquet(links_path, index=False)
        print(f"  Saved: {links_path}")

    # Save climatology
    clim_path = output_dir / f"{dataset_id}_climatology.parquet"
    climatology.to_parquet(clim_path)
    print(f"  Saved: {clim_path}")

    # Save configuration
    config_path = output_dir / f"{dataset_id}_config.yaml"
    config.to_yaml(config_path)
    print(f"  Saved: {config_path}")


def load_episode_outputs(
    dataset_id: str,
    output_dir: Optional[Path] = None,
    load_weekly_ts: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load episode analysis outputs from files.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    output_dir : Path, optional
        Output directory. If None, uses default location.
    load_weekly_ts : bool
        Whether to load weekly time series (can be large)

    Returns
    -------
    episodes : pd.DataFrame
        All episodes with features
    episode_links : pd.DataFrame
        Episode progression relationships
    climatology : pd.DataFrame
        Weekly climatology
    weekly_ts : pd.DataFrame or None
        Weekly time series (if load_weekly_ts=True)
    """
    if output_dir is None:
        output_dir = Path("./pywrdrb/episode_analysis")
    else:
        output_dir = Path(output_dir)

    # Load episodes
    episodes_path = output_dir / f"{dataset_id}_episodes_all.parquet"
    if not episodes_path.exists():
        raise FileNotFoundError(f"Episodes file not found: {episodes_path}")
    episodes = pd.read_parquet(episodes_path)
    print(f"  Loaded {len(episodes)} episodes from {episodes_path}")

    # Load links
    links_path = output_dir / f"{dataset_id}_episode_links.parquet"
    if links_path.exists():
        episode_links = pd.read_parquet(links_path)
        print(f"  Loaded {len(episode_links)} links from {links_path}")
    else:
        episode_links = pd.DataFrame()
        print(f"  No episode links file found")

    # Load climatology
    clim_path = output_dir / f"{dataset_id}_climatology.parquet"
    if clim_path.exists():
        climatology = pd.read_parquet(clim_path)
        print(f"  Loaded climatology from {clim_path}")
    else:
        climatology = pd.DataFrame()
        print(f"  No climatology file found")

    # Load weekly time series (optional, can be large)
    weekly_ts = None
    if load_weekly_ts:
        weekly_path = output_dir / f"{dataset_id}_weekly_timeseries.parquet"
        if weekly_path.exists():
            weekly_ts = pd.read_parquet(weekly_path)
            print(f"  Loaded {len(weekly_ts)} weekly records from {weekly_path}")
        else:
            print(f"  No weekly time series file found")

    return episodes, episode_links, climatology, weekly_ts


def save_analysis_results(
    comparison_results: pd.DataFrame,
    cascade_model_results: dict,
    config: 'EpisodeAnalysisConfig'
) -> None:
    """
    Save statistical analysis results.

    Parameters
    ----------
    comparison_results : pd.DataFrame
        Results from compare_episode_populations()
    cascade_model_results : dict
        Results from fit_cascade_model()
    config : EpisodeAnalysisConfig
        Configuration object
    """
    output_dir = Path(config.output_dir)
    dataset_id = config.dataset_id

    # Save comparison results
    if len(comparison_results) > 0:
        comp_path = output_dir / f"{dataset_id}_statistical_tests.csv"
        comparison_results.to_csv(comp_path, index=False)
        print(f"  Saved: {comp_path}")

    # Save model summary
    if cascade_model_results and 'error' not in cascade_model_results:
        summary_path = output_dir / f"{dataset_id}_cascade_model_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("Cascade Probability Model Summary\n")
            f.write("=" * 60 + "\n\n")

            # Model fit statistics
            stats = cascade_model_results['model_fit_stats']
            f.write("Model Fit Statistics:\n")
            f.write(f"  AIC: {stats['aic']:.2f}\n")
            f.write(f"  BIC: {stats['bic']:.2f}\n")
            f.write(f"  Pseudo R-squared: {stats['pseudo_r2']:.4f}\n")
            f.write(f"  N observations: {stats['n_observations']}\n")
            f.write(f"  N cascade: {stats['n_cascade']}\n")
            f.write(f"  N non-cascade: {stats['n_non_cascade']}\n\n")

            # Coefficients
            f.write("Coefficients:\n")
            f.write("-" * 60 + "\n")
            coef_df = cascade_model_results['coefficients']
            f.write(coef_df.to_string(index=False))
            f.write("\n")

        print(f"  Saved: {summary_path}")

        # Also save coefficients as CSV
        coef_path = output_dir / f"{dataset_id}_cascade_model_coefficients.csv"
        cascade_model_results['coefficients'].to_csv(coef_path, index=False)
        print(f"  Saved: {coef_path}")


def load_stress_episodes(
    dataset_id: str,
    output_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load only stress episodes (convenience function).

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    output_dir : Path, optional
        Output directory

    Returns
    -------
    stress_episodes : pd.DataFrame
        E1/E1d/E1c episodes with features
    """
    if output_dir is None:
        output_dir = Path("./pywrdrb/episode_analysis")
    else:
        output_dir = Path(output_dir)

    stress_path = output_dir / f"{dataset_id}_episodes_stress.parquet"
    if stress_path.exists():
        return pd.read_parquet(stress_path)

    # Fall back to loading all and filtering
    all_path = output_dir / f"{dataset_id}_episodes_all.parquet"
    if all_path.exists():
        episodes = pd.read_parquet(all_path)
        stress_types = ['E1', 'E1d', 'E1c']
        return episodes[episodes['episode_type'].isin(stress_types)]

    raise FileNotFoundError(f"No episode files found in {output_dir}")
