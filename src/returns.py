from __future__ import annotations

import numpy as np


def generate_normal_returns(
    n_simulations: int,
    n_days: int,
    mean: float,
    std: float,
    random_state: int | None = None,
) -> np.ndarray:
    """
    Generate simulated daily returns from a normal distribution.

    Returns
    -------
    np.ndarray
        Shape: (n_simulations, n_days)
    """
    rng = np.random.default_rng(random_state)
    return rng.normal(loc=mean, scale=std, size=(n_simulations, n_days))