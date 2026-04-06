# Quant Risk Simulator

A Monte Carlo risk simulation engine for analyzing portfolio outcome distributions, drawdowns, and tail risk under multiple return-generating assumptions.

---

## Overview

Traditional backtests evaluate a strategy using a single historical path. While useful, this approach hides the **distribution of possible outcomes**.

This project addresses that limitation by simulating thousands of potential portfolio paths and analyzing:

* How outcomes vary under uncertainty
* The probability of losses and large drawdowns
* Sensitivity to different return assumptions
* Differences between theoretical models and real market behavior

---

## Key Idea

Instead of asking:

> “What happened in one backtest?”

This project asks:

> “What could happen across many plausible futures?”

---

## Features

### Monte Carlo Simulation

* Simulates thousands of portfolio paths over a fixed horizon
* Models compounding effects of daily returns

### Multiple Return Models

* **Normal distribution** (baseline assumption)
* **Student-t distribution** (fat tails / extreme events)
* **Stressed volatility scenario**
* **Historical block bootstrap (SPY)** using real market data

### Risk Metrics

* Final portfolio value distribution
* Probability of loss
* Probability of ruin
* Max drawdown distribution
* Value-at-Risk (VaR) and Conditional VaR (CVaR)
* Sharpe ratio, Sortino ratio, Calmar ratio

### Visualizations

* Simulated equity curve fan plots
* Percentile cone (5th–95th range + median path)
* Final value distribution histograms
* Drawdown distribution histograms
* Cross-scenario comparison boxplots

### Structured Outputs

* Per-scenario:

  * `summary.json`
  * `path_statistics.csv`
* Global:

  * `scenario_summary.csv`
  * `scenario_comparison_paths.csv`
* Saved plots for all scenarios

---

## Project Structure

```text
quant-risk-simulator/
│
├── data/
│   └── historical_returns.csv        # SPY-based returns for bootstrap
├── results/
│   ├── baseline_normal/
│   ├── fat_tail_student_t/
│   ├── stressed_volatility/
│   ├── historical_block_bootstrap/
│   ├── scenario_summary.csv
│   └── scenario_comparison_paths.csv
│
├── src/
│   ├── config.py
│   ├── metrics.py
│   ├── plots.py
│   ├── returns.py
│   └── simulator.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## How It Works

1. Generate simulated return paths using a specified model
2. Convert returns into equity curves via compounding
3. Compute path-level statistics (returns, drawdowns, ratios)
4. Aggregate results into distributions and summary metrics
5. Visualize outcomes and compare across scenarios

---

## Example Insights

Using identical expected returns but different assumptions:

* Fat-tailed distributions produce **more extreme downside risk**
* Higher volatility significantly increases **drawdown probability**
* Historical bootstrap reveals **non-normal market behavior**
* Even with positive expected return, **loss probability remains non-trivial**

---

## Why This Matters

A single backtest can be misleading.

This project demonstrates that:

* Risk is **distributional**, not deterministic
* Tail events matter more than average outcomes
* Strategy evaluation requires **scenario-based analysis**
* Real market behavior often deviates from idealized assumptions

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run

```bash
python main.py
```

or (Windows):

```bash
py main.py
```

---

## Data

The historical bootstrap scenario uses daily SPY returns derived from price data to simulate realistic market dynamics.

If `data/historical_returns.csv` is present, the model incorporates real market behavior. Otherwise, the scenario is skipped automatically.

---

## Future Improvements

* Regime-switching return models
* Time-varying volatility
* Strategy-specific return distributions (from backtests)
* Multi-asset portfolio simulations
* Position sizing and leverage modeling

---

## Summary

This project provides a framework for understanding portfolio risk beyond single-path backtesting by combining:

* Probabilistic simulation
* Multiple return models
* Real market data

to evaluate the full distribution of possible outcomes.

---
