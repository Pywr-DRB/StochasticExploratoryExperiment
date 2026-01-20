"""
Configuration for episode-level vulnerability analysis.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import os

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Import from parent config to avoid duplication
import sys
_parent_dir = str(Path(__file__).parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from config import ROOT_DIR, PERIOD_ORIGIN
from utils import get_nyc_storage_capacities


@dataclass
class EpisodeAnalysisConfig:
    """Configuration for episode-level vulnerability analysis.

    All methodological thresholds and settings are centralized here.
    """

    # === Dataset Settings ===
    dataset_id: str = 'stationary_ensemble'

    # === Episode Identification Thresholds ===

    # E1 (Inflow Stress): inflow_std < threshold (negative = deficit)
    inflow_stress_threshold: float = -1.0

    # E1d (Demand Stress): demand_std > threshold (positive = high demand)
    demand_stress_threshold: float = 1.0

    # E1c (Combined Stress): combined_stress_std > threshold
    combined_stress_threshold: float = 1.5

    # E2 (Zone Transition): ffmp_zone > baseline_zone (higher = more severe)
    # In pywrdrb: 3=Normal, 4=Warning, 5=Watch, 6=Emergency
    baseline_ffmp_zone: int = 3

    # E3, E4 thresholds (satisfaction < threshold indicates shortage)
    satisfaction_tolerance: float = 0.999

    # === Episode Exit Criteria ===

    # Number of consecutive weeks above threshold to end episode
    exit_persistence_weeks: int = 2

    # === Episode Filtering ===

    # Minimum episode duration to include (filters transient crossings)
    min_episode_duration_weeks: int = 2

    # === Progression Linkage ===

    # Maximum lag (weeks) between episode onsets to consider "progression"
    progression_lag_window_weeks: int = 4

    # === Temporal Resolution ===

    temporal_resolution: str = "weekly"
    period_origin: str = field(default_factory=lambda: PERIOD_ORIGIN)

    # === System Constants ===
    nyc_reservoirs: List[str] = field(
        default_factory=lambda: ['cannonsville', 'pepacton', 'neversink']
    )

    @property
    def nyc_total_capacity(self) -> float:
        """Total NYC reservoir capacity in MG."""
        return get_nyc_storage_capacities()['total']

    # === Output Settings ===

    output_dir: Path = field(
        default_factory=lambda: Path(ROOT_DIR) / "pywrdrb" / "episode_analysis"
    )
    save_intermediate: bool = True
    figure_format: str = "png"
    figure_dpi: int = 300

    # === Processing Settings ===

    n_workers: int = -1  # -1 = use all available cores

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Validate thresholds
        if self.inflow_stress_threshold >= 0:
            raise ValueError("Inflow stress threshold should be negative (deficit)")
        if self.demand_stress_threshold <= 0:
            raise ValueError("Demand stress threshold should be positive (high demand)")
        if not (0 < self.satisfaction_tolerance <= 1.0):
            raise ValueError("Satisfaction tolerance must be in (0, 1]")

    @classmethod
    def from_yaml(cls, path: Path) -> 'EpisodeAnalysisConfig':
        """Load configuration from YAML file."""
        if not HAS_YAML:
            raise ImportError("PyYAML required for loading config from YAML. Install with: pip install pyyaml")
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            'dataset_id': self.dataset_id,
            'inflow_stress_threshold': self.inflow_stress_threshold,
            'demand_stress_threshold': self.demand_stress_threshold,
            'combined_stress_threshold': self.combined_stress_threshold,
            'baseline_ffmp_zone': self.baseline_ffmp_zone,
            'satisfaction_tolerance': self.satisfaction_tolerance,
            'exit_persistence_weeks': self.exit_persistence_weeks,
            'min_episode_duration_weeks': self.min_episode_duration_weeks,
            'progression_lag_window_weeks': self.progression_lag_window_weeks,
            'temporal_resolution': self.temporal_resolution,
            'period_origin': self.period_origin,
            'output_dir': str(self.output_dir),
            'save_intermediate': self.save_intermediate,
            'figure_format': self.figure_format,
            'figure_dpi': self.figure_dpi,
            'n_workers': self.n_workers,
        }
        if HAS_YAML:
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            # Fallback: write as simple key=value format
            with open(path, 'w') as f:
                for k, v in config_dict.items():
                    f.write(f"{k}: {v}\n")
