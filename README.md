# Quant Risk Simulator

A Monte Carlo risk simulation engine for analyzing strategy outcome distributions, drawdowns, and tail risk in quantitative trading.

## Overview

A single backtest shows only one realized path. This project uses Monte Carlo simulation to generate thousands of possible portfolio paths and evaluate the distribution of outcomes under uncertainty.

## Features

- Simulates daily return paths using a normal distribution
- Generates portfolio equity curves
- Computes key risk metrics:
  - final portfolio value
  - probability of loss
  - probability of ruin
  - max drawdown
  - Sharpe ratio
- Produces visualizations for:
  - simulated equity curves
  - final value distribution
  - max drawdown distribution

## Project Structure

```text
quant-risk-simulator/
│
├── data/
├── notebooks/
├── results/
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