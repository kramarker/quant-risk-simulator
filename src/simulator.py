from __future__ import annotations

import numpy as np


def simulate_equity_curves(initial_capital: float, returns: np.ndarray) -> np.ndarray:
    """
    Convert simulated return paths into equity curves.

    Parameters
    ----------
    initial_capital : float
        Starting portfolio value.
    returns : np.ndarray
        Shape: (n_simulations, n_days)

    Returns
    -------
    np.ndarray
        Equity curves with shape: (n_simulations, n_days + 1)
    """
    growth = np.cumprod(1 + returns, axis=1)
    equity_curves = initial_capital * growth

    starting_column = np.full((returns.shape[0], 1), initial_capital)
    return np.hstack([starting_column, equity_curves])