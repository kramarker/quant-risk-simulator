from __future__ import annotations

import numpy as np

from src.config import (
    INITIAL_CAPITAL,
    N_SIMULATIONS,
    N_DAYS,
    MEAN_DAILY_RETURN,
    DAILY_VOLATILITY,
    RISK_FREE_RATE,
    PERIODS_PER_YEAR,
    N_PLOT_PATHS,
    RANDOM_STATE,
    RUIN_THRESHOLD,
)
from src.returns import generate_normal_returns
from src.simulator import simulate_equity_curves
from src.metrics import summarize_simulations, compute_max_drawdown
from src.plots import (
    plot_equity_curves,
    plot_final_value_histogram,
    plot_drawdown_histogram,
)


def main() -> None:
    simulated_returns = generate_normal_returns(
        n_simulations=N_SIMULATIONS,
        n_days=N_DAYS,
        mean=MEAN_DAILY_RETURN,
        std=DAILY_VOLATILITY,
        random_state=RANDOM_STATE,
    )

    equity_curves = simulate_equity_curves(
        initial_capital=INITIAL_CAPITAL,
        returns=simulated_returns,
    )

    summary = summarize_simulations(
        equity_curves=equity_curves,
        returns=simulated_returns,
        initial_capital=INITIAL_CAPITAL,
        ruin_threshold=RUIN_THRESHOLD,
        risk_free_rate=RISK_FREE_RATE,
        periods_per_year=PERIODS_PER_YEAR,
    )

    final_values = equity_curves[:, -1]
    max_drawdowns = np.array([compute_max_drawdown(curve) for curve in equity_curves])

    print("\nMonte Carlo Risk Simulation Summary")
    print("-" * 40)
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")

    plot_equity_curves(
        equity_curves=equity_curves,
        n_paths=N_PLOT_PATHS,
        save_path="results/equity_curves.png",
    )

    plot_final_value_histogram(
        final_values=final_values,
        save_path="results/final_value_histogram.png",
    )

    plot_drawdown_histogram(
        max_drawdowns=max_drawdowns,
        save_path="results/max_drawdown_histogram.png",
    )


if __name__ == "__main__":
    main()