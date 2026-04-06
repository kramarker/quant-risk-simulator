from __future__ import annotations

import numpy as np
import pandas as pd


def compute_drawdown_series(equity_curve: np.ndarray) -> np.ndarray:
    running_peak = np.maximum.accumulate(equity_curve)
    return (equity_curve - running_peak) / running_peak


def compute_max_drawdown(equity_curve: np.ndarray) -> float:
    return float(np.min(compute_drawdown_series(equity_curve)))


def compute_annualized_return(returns: np.ndarray, periods_per_year: int = 252) -> float:
    n = len(returns)
    if n == 0:
        return 0.0
    total_growth = np.prod(1 + returns)
    return float(total_growth ** (periods_per_year / n) - 1)


def compute_annualized_volatility(returns: np.ndarray, periods_per_year: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    return float(np.std(returns, ddof=1) * np.sqrt(periods_per_year))


def compute_downside_volatility(
    returns: np.ndarray,
    target_return: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    downside = np.minimum(returns - target_return, 0.0)
    if len(downside) < 2:
        return 0.0
    return float(np.std(downside, ddof=1) * np.sqrt(periods_per_year))


def compute_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns - (risk_free_rate / periods_per_year)
    vol = np.std(excess, ddof=1)
    if vol == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * np.mean(excess) / vol)


def compute_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    annual_return = compute_annualized_return(returns, periods_per_year=periods_per_year)
    downside_vol = compute_downside_volatility(
        returns,
        target_return=risk_free_rate / periods_per_year,
        periods_per_year=periods_per_year,
    )
    if downside_vol == 0:
        return 0.0
    return float((annual_return - risk_free_rate) / downside_vol)


def compute_calmar_ratio(
    returns: np.ndarray,
    equity_curve: np.ndarray,
    periods_per_year: int = 252,
) -> float:
    annual_return = compute_annualized_return(returns, periods_per_year=periods_per_year)
    max_dd = abs(compute_max_drawdown(equity_curve))
    if max_dd == 0:
        return 0.0
    return float(annual_return / max_dd)


def compute_var_cvar(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    var = float(np.quantile(values, alpha))
    tail = values[values <= var]
    cvar = float(np.mean(tail)) if len(tail) > 0 else var
    return var, cvar


def build_path_statistics(
    equity_curves: np.ndarray,
    returns: np.ndarray,
    initial_capital: float,
    risk_free_rate: float,
    periods_per_year: int,
) -> pd.DataFrame:
    rows = []

    for i in range(len(equity_curves)):
        curve = equity_curves[i]
        path_returns = returns[i]

        final_value = float(curve[-1])
        terminal_return = float(final_value / initial_capital - 1.0)
        annual_return = compute_annualized_return(path_returns, periods_per_year=periods_per_year)
        annual_vol = compute_annualized_volatility(path_returns, periods_per_year=periods_per_year)
        max_dd = compute_max_drawdown(curve)
        sharpe = compute_sharpe_ratio(
            path_returns,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )
        sortino = compute_sortino_ratio(
            path_returns,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )
        calmar = compute_calmar_ratio(
            path_returns,
            curve,
            periods_per_year=periods_per_year,
        )

        rows.append(
            {
                "path_id": i,
                "final_value": final_value,
                "terminal_return": terminal_return,
                "annualized_return": annual_return,
                "annualized_volatility": annual_vol,
                "max_drawdown": max_dd,
                "sharpe_ratio": sharpe,
                "sortino_ratio": sortino,
                "calmar_ratio": calmar,
            }
        )

    return pd.DataFrame(rows)


def summarize_simulations(
    path_stats: pd.DataFrame,
    initial_capital: float,
    ruin_threshold: float,
) -> dict[str, float]:
    final_values = path_stats["final_value"].to_numpy()
    terminal_returns = path_stats["terminal_return"].to_numpy()
    max_drawdowns = path_stats["max_drawdown"].to_numpy()

    ruin_level = initial_capital * ruin_threshold

    var_5, cvar_5 = compute_var_cvar(terminal_returns, alpha=0.05)
    var_1, cvar_1 = compute_var_cvar(terminal_returns, alpha=0.01)

    summary = {
        "mean_final_value": float(np.mean(final_values)),
        "median_final_value": float(np.median(final_values)),
        "min_final_value": float(np.min(final_values)),
        "max_final_value": float(np.max(final_values)),
        "5th_percentile_final_value": float(np.percentile(final_values, 5)),
        "95th_percentile_final_value": float(np.percentile(final_values, 95)),
        "probability_of_loss": float(np.mean(final_values < initial_capital)),
        "probability_of_ruin": float(np.mean(final_values < ruin_level)),
        "probability_of_20pct_drawdown": float(np.mean(max_drawdowns <= -0.20)),
        "probability_of_30pct_drawdown": float(np.mean(max_drawdowns <= -0.30)),
        "average_terminal_return": float(np.mean(terminal_returns)),
        "median_terminal_return": float(np.median(terminal_returns)),
        "var_5_terminal_return": var_5,
        "cvar_5_terminal_return": cvar_5,
        "var_1_terminal_return": var_1,
        "cvar_1_terminal_return": cvar_1,
        "average_annualized_return": float(path_stats["annualized_return"].mean()),
        "average_annualized_volatility": float(path_stats["annualized_volatility"].mean()),
        "average_max_drawdown": float(path_stats["max_drawdown"].mean()),
        "worst_max_drawdown": float(path_stats["max_drawdown"].min()),
        "average_sharpe_ratio": float(path_stats["sharpe_ratio"].mean()),
        "average_sortino_ratio": float(path_stats["sortino_ratio"].mean()),
        "average_calmar_ratio": float(path_stats["calmar_ratio"].mean()),
    }

    return summary