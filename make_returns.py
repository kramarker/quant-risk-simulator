import pandas as pd
from pathlib import Path

source = Path("../quant-trading-backtester/data/SPY_data.csv")
target = Path("data/historical_returns.csv")

df = pd.read_csv(source)

price_col = None
for col in ["Adj Close", "Close", "close", "adj_close"]:
    if col in df.columns:
        price_col = col
        break

if price_col is None:
    raise ValueError(f"No valid price column found. Columns were: {list(df.columns)}")

returns = df[[price_col]].copy()
returns["return"] = returns[price_col].pct_change()
returns = returns[["return"]].dropna()

target.parent.mkdir(parents=True, exist_ok=True)
returns.to_csv(target, index=False)

print(f"Saved {len(returns)} rows to {target}")
print(returns.head())