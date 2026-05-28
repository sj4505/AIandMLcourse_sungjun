"""
Hooke's Law Dataset
F = kx  →  Length = 2 * Weight + 10
(k = 2 cm/kg, initial length = 10 cm)
"""
import numpy as np
from typing import Tuple


def get_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (weights, measured_lengths, true_lengths) — same data as week2."""
    weights = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    true_lengths = 2.0 * weights + 10.0
    np.random.seed(42)
    noise = np.random.normal(loc=0.0, scale=1.5, size=len(weights))
    measured_lengths = true_lengths + noise
    return weights, measured_lengths, true_lengths


def get_spring_constant() -> float:
    """True spring constant k (cm/kg)."""
    return 2.0


def get_initial_length() -> float:
    """Natural length of the spring (cm)."""
    return 10.0
