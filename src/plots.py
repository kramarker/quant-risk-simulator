from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _finalize_plot(save_path: str | None, show: bool) -> None:
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_equity_curves(
    equity_curves: np.ndarray,
    title: str,
    n_paths: int = 100,
    save_path: str | None = None,
    show: bool = False,
) -> None:
    n_paths = min(n_paths, equity_curves.shape[0])

    plt.figure(figsize=(11, 6))
    for i in range(n_paths):
        plt.plot(equity_curves[i], alpha=0.18, linewidth=1)

    plt.title(title)
    plt.xlabel("Day")
    plt.ylabel("Portfolio Value")
    plt.tight_layout()
    _finalize_plot(save_path, show)


def plot_percentile_cone(
    equity_curves: np.ndarray,
    title: str,
    save_path: str | None = None,
    show: bool = False,
) -> None:
    days = np.arange(equity_curves.shape[1])

    p5 = np.percentile(equity_curves, 5, axis=0)
    p25 = np.percentile(equity_curves, 25, axis=0)
    p50 = np.percentile(equity_curves, 50, axis=0)
    p75 = np.percentile(equity_curves, 75, axis=0)
    p95 = np.percentile(equity_curves, 95, axis=0)

    plt.figure(figsize=(11, 6))
    plt.fill_between(days, p5, p95, alpha=0.20, label="5th-95th percentile")
    plt.fill_between(days, p25, p75, alpha=0.35, label="25th-75th percentile")
    plt.plot(days, p50, linewidth=2.0, label="Median path")

    plt.title(title)
    plt.xlabel("Day")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.tight_layout()
    _finalize_plot(save_path, show)


def plot_final_value_histogram(
    final_values: np.ndarray,
    title: str,
    save_path: str | None = None,
    show: bool = False,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(final_values, bins=50)
    plt.title(title)
    plt.xlabel("Final Portfolio Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    _finalize_plot(save_path, show)


def plot_drawdown_histogram(
    max_drawdowns: np.ndarray,
    title: str,
    save_path: str | None = None,
    show: bool = False,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(max_drawdowns, bins=50)
    plt.title(title)
    plt.xlabel("Max Drawdown")
    plt.ylabel("Frequency")
    plt.tight_layout()
    _finalize_plot(save_path, show)


def plot_scenario_final_value_boxplot(
    comparison_df: pd.DataFrame,
    save_path: str | None = None,
    show: bool = False,
) -> None:
    grouped = [
        comparison_df.loc[comparison_df["scenario"] == scenario, "final_value"].to_numpy()
        for scenario in comparison_df["scenario"].unique()
    ]
    labels = list(comparison_df["scenario"].unique())

    plt.figure(figsize=(12, 6))
    plt.boxplot(grouped, tick_labels=labels)
    plt.title("Final Portfolio Value by Scenario")
    plt.xlabel("Scenario")
    plt.ylabel("Final Portfolio Value")
    plt.xticks(rotation=15)
    plt.tight_layout()
    _finalize_plot(save_path, show)


def plot_scenario_drawdown_boxplot(
    comparison_df: pd.DataFrame,
    save_path: str | None = None,
    show: bool = False,
) -> None:
    grouped = [
        comparison_df.loc[comparison_df["scenario"] == scenario, "max_drawdown"].to_numpy()
        for scenario in comparison_df["scenario"].unique()
    ]
    labels = list(comparison_df["scenario"].unique())

    plt.figure(figsize=(12, 6))
    plt.boxplot(grouped, tick_labels=labels)
    plt.title("Max Drawdown by Scenario")
    plt.xlabel("Scenario")
    plt.ylabel("Max Drawdown")
    plt.xticks(rotation=15)
    plt.tight_layout()
    _finalize_plot(save_path, show)