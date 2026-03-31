from __future__ import annotations

INITIAL_CAPITAL = 10_000
N_SIMULATIONS = 5_000
N_DAYS = 252

RISK_FREE_RATE = 0.02
PERIODS_PER_YEAR = 252
RUIN_THRESHOLD = 0.50
PLOT_PATHS = 100
RANDOM_STATE = 42

SHOW_PLOTS = False
SAVE_PLOTS = True

HISTORICAL_RETURNS_PATH = "data/historical_returns.csv"

SCENARIOS = [
    {
        "name": "baseline_normal",
        "model": "normal",
        "mean": 0.0005,
        "std": 0.01,
    },
    {
        "name": "fat_tail_student_t",
        "model": "student_t",
        "mean": 0.0005,
        "std": 0.01,
        "df": 5,
    },
    {
        "name": "stressed_volatility",
        "model": "normal",
        "mean": 0.0003,
        "std": 0.018,
    },
    {
        "name": "historical_block_bootstrap",
        "model": "block_bootstrap",
        "block_size": 5,
    },
]