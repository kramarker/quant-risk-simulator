from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def generate_normal_returns(
    n_simulations: int,
    n_days: int,
    mean: float,
    std: float,
    random_state: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    return rng.normal(loc=mean, scale=std, size=(n_simulations, n_days))


def generate_student_t_returns(
    n_simulations: int,
    n_days: int,
    mean: float,
    std: float,
    df: int,
    random_state: int | None = None,
) -> np.ndarray:
    if df <= 2:
        raise ValueError("df must be > 2 for finite variance.")

    rng = np.random.default_rng(random_state)
    raw = rng.standard_t(df=df, size=(n_simulations, n_days))

    # Standardize Student-t to unit variance, then scale to desired std
    standardized = raw / np.sqrt(df / (df - 2))
    return mean + std * standardized


def load_historical_returns(csv_path: str) -> np.ndarray:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Historical returns file not found: {csv_path}")

    df = pd.read_csv(path)

    possible_return_cols = ["return", "returns", "daily_return", "pct_return"]
    for col in possible_return_cols:
        if col in df.columns:
            returns = df[col].dropna().to_numpy(dtype=float)
            if len(returns) == 0:
                raise ValueError("Historical return column exists but is empty.")
            return returns

    possible_price_cols = [
        "Adj Close",
        "adj_close",
        "adjusted_close",
        "Close",
        "close",
        "price",
    ]
    for col in possible_price_cols:
        if col in df.columns:
            prices = df[col].dropna().to_numpy(dtype=float)
            if len(prices) < 2:
                raise ValueError("Need at least 2 price observations to compute returns.")
            returns = pd.Series(prices).pct_change().dropna().to_numpy(dtype=float)
            return returns

    raise ValueError(
        "CSV must contain either a return column "
        "(return, returns, daily_return, pct_return) "
        "or a price column (Close, Adj Close, close, adj_close, price)."
    )


def bootstrap_iid_returns(
    historical_returns: np.ndarray,
    n_simulations: int,
    n_days: int,
    random_state: int | None = None,
) -> np.ndarray:
    if len(historical_returns) == 0:
        raise ValueError("historical_returns is empty.")

    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, len(historical_returns), size=(n_simulations, n_days))
    return historical_returns[indices]


def bootstrap_block_returns(
    historical_returns: np.ndarray,
    n_simulations: int,
    n_days: int,
    block_size: int = 5,
    random_state: int | None = None,
) -> np.ndarray:
    if len(historical_returns) < block_size:
        raise ValueError("historical_returns length must be >= block_size.")

    rng = np.random.default_rng(random_state)
    max_start = len(historical_returns) - block_size
    paths = np.empty((n_simulations, n_days), dtype=float)

    for i in range(n_simulations):
        path = []
        while len(path) < n_days:
            start = rng.integers(0, max_start + 1)
            block = historical_returns[start : start + block_size]
            path.extend(block.tolist())
        paths[i] = np.array(path[:n_days], dtype=float)

    return paths