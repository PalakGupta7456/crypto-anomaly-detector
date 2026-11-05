"""Synthetic candles with the live schema (offline runs and tests).

Prices follow GARCH-like volatility clustering, volume co-moves with absolute
returns, and a few genuine shocks are planted so the dashboard has events.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import COINS, DATA_DIR

START_PRICE = {"BTC": 68000, "ETH": 3400, "SOL": 160, "XRP": 0.6, "ADA": 0.45, "DOGE": 0.15, "AVAX": 35, "LINK": 15}


def generate(days: int = 90, seed: int = 3, end: pd.Timestamp | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = end or pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")
    times = pd.date_range(end - pd.Timedelta(days=days), end, freq="h")
    n = len(times)
    hour = times.hour.values
    market = rng.standard_t(5, n) * 0.004
    frames = []
    for sym in COINS:
        beta = 1.0 if sym == "BTC" else rng.uniform(1.1, 1.6)
        vol = np.empty(n)
        r = np.empty(n)
        v = 0.006
        for i in range(n):
            v = np.sqrt(2e-6 + 0.06 * (r[i - 1] if i else 0) ** 2 + 0.9 * v**2)
            r[i] = beta * market[i] + v * rng.standard_t(4) / np.sqrt(2)
            vol[i] = v
        base_volume = 1e6 / START_PRICE[sym] * (1 + 0.3 * np.sin(2 * np.pi * (hour - 15) / 24))
        volume = base_volume * np.exp(np.clip(25 * np.abs(r), 0, 5) + rng.normal(0, 0.25, n))
        for j in rng.choice(np.arange(200, n - 5), 3, replace=False):  # planted shocks
            if rng.random() < 0.5:
                r[j] += rng.choice([-1, 1]) * rng.uniform(0.05, 0.09)
            volume[j] *= rng.uniform(6, 12)
        close = START_PRICE[sym] * np.exp(np.cumsum(r))
        open_ = np.r_[close[0], close[:-1]]
        wick = np.abs(rng.normal(0, 1, n)) * vol * close
        frames.append(pd.DataFrame({
            "time": times, "low": np.minimum(open_, close) - wick, "high": np.maximum(open_, close) + wick,
            "open": open_, "close": close, "volume": volume, "coin": sym,
        }))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    generate().to_parquet(DATA_DIR / "candles.parquet", index=False)
    print("Synthetic candles written")


if __name__ == "__main__":
    main()
