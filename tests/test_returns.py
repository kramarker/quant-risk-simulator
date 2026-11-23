import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.returns import (
    bootstrap_block_returns,
    bootstrap_iid_returns,
    generate_normal_returns,
    generate_student_t_returns,
    load_historical_returns,
)


def test_normal_returns_shape_and_moments():
    r = generate_normal_returns(2_000, 252, mean=0.0005, std=0.01, random_state=1)
    assert r.shape == (2_000, 252)
    assert r.mean() == pytest.approx(0.0005, abs=2e-4)
    assert r.std() == pytest.approx(0.01, rel=0.02)


def test_same_seed_reproduces_paths_and_different_seed_does_not():
    a = generate_normal_returns(10, 20, 0.0, 0.01, random_state=42)
    b = generate_normal_returns(10, 20, 0.0, 0.01, random_state=42)
    c = generate_normal_returns(10, 20, 0.0, 0.01, random_state=43)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_student_t_is_scaled_to_target_volatility():
    # The Student-t scenario is meant to differ from the Gaussian only in tail shape,
    # so its standard deviation must match the requested std, not the raw t variance df/(df-2).
    r = generate_student_t_returns(5_000, 252, mean=0.0005, std=0.01, df=5, random_state=7)
    assert r.std() == pytest.approx(0.01, rel=0.03)
    assert r.mean() == pytest.approx(0.0005, abs=2e-4)


def test_student_t_has_heavier_tails_than_normal():
    t = generate_student_t_returns(2_000, 252, 0.0, 0.01, df=5, random_state=3).ravel()
    n = generate_normal_returns(2_000, 252, 0.0, 0.01, random_state=3).ravel()
    assert stats.kurtosis(t) > stats.kurtosis(n) + 1.0


def test_student_t_rejects_infinite_variance():
    with pytest.raises(ValueError):
        generate_student_t_returns(10, 10, 0.0, 0.01, df=2)


def test_iid_bootstrap_only_uses_historical_values():
    hist = np.array([0.01, -0.02, 0.005, 0.0])
    r = bootstrap_iid_returns(hist, 50, 30, random_state=0)
    assert r.shape == (50, 30)
    assert set(np.unique(r)).issubset(set(hist))


def test_block_bootstrap_keeps_consecutive_blocks():
    hist = np.arange(100, dtype=float) / 1000  # strictly increasing, so blocks are easy to recognize
    r = bootstrap_block_returns(hist, n_simulations=20, n_days=25, block_size=5, random_state=0)
    assert r.shape == (20, 25)
    for path in r:
        for start in range(0, 25, 5):
            block = path[start:start + 5]
            # each block is a run of consecutive historical observations
            np.testing.assert_allclose(np.diff(block), 0.001)


def test_block_bootstrap_rejects_short_history():
    with pytest.raises(ValueError):
        bootstrap_block_returns(np.array([0.01, 0.02]), 5, 10, block_size=5)


def test_load_historical_returns_from_return_column(tmp_path):
    path = tmp_path / "returns.csv"
    pd.DataFrame({"return": [0.01, None, -0.02]}).to_csv(path, index=False)
    np.testing.assert_allclose(load_historical_returns(str(path)), [0.01, -0.02])


def test_load_historical_returns_from_prices(tmp_path):
    path = tmp_path / "prices.csv"
    pd.DataFrame({"Close": [100.0, 110.0, 99.0]}).to_csv(path, index=False)
    np.testing.assert_allclose(load_historical_returns(str(path)), [0.10, -0.10])


def test_load_historical_returns_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_historical_returns(str(tmp_path / "missing.csv"))
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"volume": [1, 2, 3]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        load_historical_returns(str(bad))
