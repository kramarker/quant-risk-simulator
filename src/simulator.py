from __future__ import annotations

import numpy as np


def simulate_equity_curves(initial_capital: float, returns: np.ndarray) -> np.ndarray:
    growth = np.cumprod(1 + returns, axis=1)
    equity = initial_capital * growth
    starting_col = np.full((returns.shape[0], 1), initial_capital, dtype=float)
    return np.hstack([starting_col, equity])


def compute_terminal_returns(equity_curves: np.ndarray, initial_capital: float) -> np.ndarray:
    return (equity_curves[:, -1] / initial_capital) - 1.0