from __future__ import annotations

import numpy as np


def compute_max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Compute max drawdown for a single equity curve.
    """
    running_peak = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_peak) / running_peak
    return float(np.min(drawdowns))


def compute_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Compute annualized Sharpe ratio for a single return series.
    """
    excess_returns = returns - (risk_free_rate / periods_per_year)
    std = np.std(excess_returns, ddof=1)

    if std == 0:
        return 0.0

    return float(np.sqrt(periods_per_year) * np.mean(excess_returns) / std)


def summarize_simulations(
    equity_curves: np.ndarray,
    returns: np.ndarray,
    initial_capital: float,
    ruin_threshold: float,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """
    Summarize Monte Carlo simulation results.
    """
    final_values = equity_curves[:, -1]
    max_drawdowns = np.array([compute_max_drawdown(curve) for curve in equity_curves])
    sharpes = np.array(
        [
            compute_sharpe_ratio(path, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year)
            for path in returns
        ]
    )

    ruin_level = initial_capital * ruin_threshold

    summary = {
        "mean_final_value": float(np.mean(final_values)),
        "median_final_value": float(np.median(final_values)),
        "5th_percentile_final_value": float(np.percentile(final_values, 5)),
        "95th_percentile_final_value": float(np.percentile(final_values, 95)),
        "probability_of_loss": float(np.mean(final_values < initial_capital)),
        "probability_of_ruin": float(np.mean(final_values < ruin_level)),
        "average_max_drawdown": float(np.mean(max_drawdowns)),
        "worst_max_drawdown": float(np.min(max_drawdowns)),
        "average_sharpe_ratio": float(np.mean(sharpes)),
    }

    return summary