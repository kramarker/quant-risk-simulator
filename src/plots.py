from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_equity_curves(equity_curves: np.ndarray, n_paths: int = 100, save_path: str | None = None) -> None:
    """
    Plot a subset of simulated equity curves.
    """
    n_paths = min(n_paths, equity_curves.shape[0])

    plt.figure(figsize=(10, 6))
    for i in range(n_paths):
        plt.plot(equity_curves[i], alpha=0.35)

    plt.title("Monte Carlo Simulated Equity Curves")
    plt.xlabel("Day")
    plt.ylabel("Portfolio Value")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_final_value_histogram(final_values: np.ndarray, save_path: str | None = None) -> None:
    """
    Plot histogram of ending portfolio values.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(final_values, bins=50)
    plt.title("Distribution of Final Portfolio Values")
    plt.xlabel("Final Portfolio Value")
    plt.ylabel("Frequency")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_drawdown_histogram(max_drawdowns: np.ndarray, save_path: str | None = None) -> None:
    """
    Plot histogram of max drawdowns.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(max_drawdowns, bins=50)
    plt.title("Distribution of Max Drawdowns")
    plt.xlabel("Max Drawdown")
    plt.ylabel("Frequency")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()