from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

from src.config import (
    INITIAL_CAPITAL,
    N_SIMULATIONS,
    N_DAYS,
    RISK_FREE_RATE,
    PERIODS_PER_YEAR,
    RUIN_THRESHOLD,
    PLOT_PATHS,
    RANDOM_STATE,
    SHOW_PLOTS,
    SAVE_PLOTS,
    HISTORICAL_RETURNS_PATH,
    SCENARIOS,
)
from src.returns import (
    generate_normal_returns,
    generate_student_t_returns,
    load_historical_returns,
    bootstrap_block_returns,
    bootstrap_iid_returns,
)
from src.simulator import simulate_equity_curves
from src.metrics import build_path_statistics, summarize_simulations
from src.plots import (
    plot_equity_curves,
    plot_percentile_cone,
    plot_final_value_histogram,
    plot_drawdown_histogram,
    plot_scenario_final_value_boxplot,
    plot_scenario_drawdown_boxplot,
)


def get_returns_for_scenario(
    scenario: dict,
    scenario_seed: int | None,
) -> tuple[pd.DataFrame | None, object]:
    model = scenario["model"]

    if model == "normal":
        returns = generate_normal_returns(
            n_simulations=N_SIMULATIONS,
            n_days=N_DAYS,
            mean=scenario["mean"],
            std=scenario["std"],
            random_state=scenario_seed,
        )
        return None, returns

    if model == "student_t":
        returns = generate_student_t_returns(
            n_simulations=N_SIMULATIONS,
            n_days=N_DAYS,
            mean=scenario["mean"],
            std=scenario["std"],
            df=scenario["df"],
            random_state=scenario_seed,
        )
        return None, returns

    if model in {"iid_bootstrap", "block_bootstrap"}:
        historical = load_historical_returns(HISTORICAL_RETURNS_PATH)
        if model == "iid_bootstrap":
            returns = bootstrap_iid_returns(
                historical_returns=historical,
                n_simulations=N_SIMULATIONS,
                n_days=N_DAYS,
                random_state=scenario_seed,
            )
        else:
            returns = bootstrap_block_returns(
                historical_returns=historical,
                n_simulations=N_SIMULATIONS,
                n_days=N_DAYS,
                block_size=scenario.get("block_size", 5),
                random_state=scenario_seed,
            )
        return pd.DataFrame({"historical_return": historical}), returns

    raise ValueError(f"Unsupported model: {model}")


def save_summary_json(summary: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)


def print_summary(name: str, summary: dict) -> None:
    print(f"\nScenario: {name}")
    print("-" * 60)
    for key, value in summary.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


def main() -> None:
    results_root = Path("results")
    results_root.mkdir(exist_ok=True)

    summary_rows: list[dict] = []
    comparison_frames: list[pd.DataFrame] = []

    for i, scenario in enumerate(SCENARIOS):
        scenario_name = scenario["name"]
        scenario_seed = None if RANDOM_STATE is None else RANDOM_STATE + i
        scenario_dir = results_root / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        try:
            _, simulated_returns = get_returns_for_scenario(scenario, scenario_seed)
        except FileNotFoundError:
            print(
                f"\nSkipping scenario '{scenario_name}' because "
                f"{HISTORICAL_RETURNS_PATH} was not found."
            )
            continue

        equity_curves = simulate_equity_curves(
            initial_capital=INITIAL_CAPITAL,
            returns=simulated_returns,
        )

        path_stats = build_path_statistics(
            equity_curves=equity_curves,
            returns=simulated_returns,
            initial_capital=INITIAL_CAPITAL,
            risk_free_rate=RISK_FREE_RATE,
            periods_per_year=PERIODS_PER_YEAR,
        )

        summary = summarize_simulations(
            path_stats=path_stats,
            initial_capital=INITIAL_CAPITAL,
            ruin_threshold=RUIN_THRESHOLD,
        )
        summary["scenario"] = scenario_name

        print_summary(scenario_name, summary)

        path_stats.to_csv(scenario_dir / "path_statistics.csv", index=False)
        save_summary_json(summary, scenario_dir / "summary.json")

        final_values = path_stats["final_value"].to_numpy()
        max_drawdowns = path_stats["max_drawdown"].to_numpy()

        save_curves = str(scenario_dir / "equity_curves.png") if SAVE_PLOTS else None
        save_cone = str(scenario_dir / "percentile_cone.png") if SAVE_PLOTS else None
        save_final_hist = str(scenario_dir / "final_value_histogram.png") if SAVE_PLOTS else None
        save_dd_hist = str(scenario_dir / "drawdown_histogram.png") if SAVE_PLOTS else None

        plot_equity_curves(
            equity_curves=equity_curves,
            title=f"{scenario_name}: Simulated Equity Curves",
            n_paths=PLOT_PATHS,
            save_path=save_curves,
            show=SHOW_PLOTS,
        )

        plot_percentile_cone(
            equity_curves=equity_curves,
            title=f"{scenario_name}: Percentile Cone",
            save_path=save_cone,
            show=SHOW_PLOTS,
        )

        plot_final_value_histogram(
            final_values=final_values,
            title=f"{scenario_name}: Final Portfolio Value Distribution",
            save_path=save_final_hist,
            show=SHOW_PLOTS,
        )

        plot_drawdown_histogram(
            max_drawdowns=max_drawdowns,
            title=f"{scenario_name}: Max Drawdown Distribution",
            save_path=save_dd_hist,
            show=SHOW_PLOTS,
        )

        summary_rows.append(summary)

        temp = path_stats.copy()
        temp["scenario"] = scenario_name
        comparison_frames.append(temp)

    if not summary_rows:
        print("\nNo scenarios were run.")
        return

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_root / "scenario_summary.csv", index=False)

    comparison_df = pd.concat(comparison_frames, ignore_index=True)
    comparison_df.to_csv(results_root / "scenario_comparison_paths.csv", index=False)

    plot_scenario_final_value_boxplot(
        comparison_df=comparison_df,
        save_path=str(results_root / "scenario_final_value_boxplot.png") if SAVE_PLOTS else None,
        show=SHOW_PLOTS,
    )

    plot_scenario_drawdown_boxplot(
        comparison_df=comparison_df,
        save_path=str(results_root / "scenario_drawdown_boxplot.png") if SAVE_PLOTS else None,
        show=SHOW_PLOTS,
    )

    print("\nSaved outputs:")
    print("- results/scenario_summary.csv")
    print("- results/scenario_comparison_paths.csv")
    print("- results/<scenario_name>/summary.json")
    print("- results/<scenario_name>/path_statistics.csv")
    print("- results/<scenario_name>/*.png")
    print("- results/scenario_final_value_boxplot.png")
    print("- results/scenario_drawdown_boxplot.png")


if __name__ == "__main__":
    main()