"""some utils for testing"""

from typing import Dict, Tuple

Ensemble = Tuple[str, Dict[str, float], str, Dict[str, float]]

std_normal_ensemble: Ensemble = (
    "normal",
    {"mu": 0.0, "std": 1.0},
    "normal",
    {"mu": 0.0, "std": 1.0},
)

uniform_ensemble: Ensemble = (
    "uniform",
    {"lb": 0.5, "ub": 1.5},
    "uniform",
    {"lb": 0.5, "ub": 1.5},
)

discrete_ensemble: Ensemble = (
    "discrete",
    {},
    "discrete",
    {},
)
