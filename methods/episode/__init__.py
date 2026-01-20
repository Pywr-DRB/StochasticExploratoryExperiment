"""
Episode-level vulnerability analysis for the Delaware River Basin.

This subpackage provides tools for identifying, characterizing, and analyzing
stress episodes across stochastic ensemble simulations of the NYC reservoir system.
"""

from .config import EpisodeAnalysisConfig
from .identification import (
    identify_episodes,
    identify_all_episodes,
)
from .characterization import characterize_episodes
from .linkage import link_episodes
from .analysis import (
    compare_episode_populations,
    fit_cascade_model,
)
from .io import (
    save_episode_outputs,
    load_episode_outputs,
)

__all__ = [
    'EpisodeAnalysisConfig',
    'identify_episodes',
    'identify_all_episodes',
    'characterize_episodes',
    'link_episodes',
    'compare_episode_populations',
    'fit_cascade_model',
    'save_episode_outputs',
    'load_episode_outputs',
]
