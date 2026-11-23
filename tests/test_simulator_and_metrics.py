import numpy as np
import pandas as pd
import pytest

from src import config
from src.metrics import (
    build_path_statistics,
    compute_annualized_return,
    compute_max_drawdown,
    compute_sharpe_ratio,
    compute_var_cvar,
    summarize_simulations,
)
from src.simulator import compute_terminal_returns, simulate_equity_curves


def test_equity_curves_start_at_initial_capital_and_compound():
    returns = np.array([[0.10, -0.50], [0.0, 0.0]])
    curves = simulate_equity_curves(1_000.0, returns)
    assert curves.shape == (2, 3)
    np.testing.assert_allclose(curves[0], [1_000.0, 1_100.0, 550.0])
    np.testing.assert_allclose(curves[1], [1_000.0, 1_000.0, 1_000.0])
    np.testing.assert_allclose(compute_terminal_returns(curves, 1_000.0), [-0.45, 0.0])


def test_max_drawdown_on_known_curve():
    assert compute_max_drawdown(np.array([100.0, 120.0, 90.0, 130.0])) == pytest.approx(-0.25)


def test_var_and_cvar_on_known_distribution():
    values = np.arange(1, 101, dtype=float)  # 1..100
    var, cvar = compute_var_cvar(values, alpha=0.05)
    assert var == pytest.approx(np.quantile(values, 0.05))
    assert cvar == pytest.approx(values[values <= var].mean())
    assert cvar <= var


def test_annualized_return_and_zero_vol_sharpe():
    daily = np.full(252, 0.001)
    assert compute_annualized_return(daily) == pytest.approx(1.001 ** 252 - 1)
    assert compute_sharpe_ratio(np.zeros(10)) == 0.0


def test_summary_probabilities_are_consistent():
    curves = simulate_equity_curves(100.0, np.array([[0.1], [-0.1], [-0.6], [0.0]]))
    stats = build_path_statistics(curves, np.array([[0.1], [-0.1], [-0.6], [0.0]]), 100.0, 0.0, 252)
    summary = summarize_simulations(stats, initial_capital=100.0, ruin_threshold=0.5)
    assert summary["probability_of_loss"] == pytest.approx(0.5)
    assert summary["probability_of_ruin"] == pytest.approx(0.25)
    assert summary["cvar_1_terminal_return"] <= summary["var_1_terminal_return"]


def test_config_matches_documented_design():
    assert config.N_SIMULATIONS == 5_000
    assert config.N_DAYS == 252
    scenarios = {s["name"]: s for s in config.SCENARIOS}
    # Student-t is compared with the Gaussian baseline at equal mean and volatility.
    assert scenarios["fat_tail_student_t"]["std"] == scenarios["baseline_normal"]["std"]
    assert scenarios["fat_tail_student_t"]["mean"] == scenarios["baseline_normal"]["mean"]
    assert scenarios["historical_block_bootstrap"]["block_size"] == 5
