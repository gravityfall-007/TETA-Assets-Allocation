"""TETA package exports for convenience."""

from .utils import compute_daily_returns, compute_annualized_stats
from .teta_optimizer import TETA_Optimizer

__all__ = [
    "compute_daily_returns",
    "compute_annualized_stats",
    "TETA_Optimizer",
]
