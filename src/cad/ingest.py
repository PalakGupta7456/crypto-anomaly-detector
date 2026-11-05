"""Download hourly OHLCV candles from the public Coinbase Exchange API."""
from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta

import pandas as pd
import requests

from .config import COINS, DATA_DIR, GRANULARITY, HISTORY_DAYS

log = logging.getLogger(__name__)
URL = "https://api.exchange.coinbase.com/products/{pid}/candles"
MAX_CANDLES = 300  # API limit per request


def _get(url: str, params: dict, retries: int = 5) -> list:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30, headers={"User-Agent": "crypto-anomaly-detector"})
            if r.status_code == 429:
                raise requests.HTTPError("rate limited")
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:  # pragma: no cover - network
            wait = 1.5 * 2**attempt
            log.warning("%s, retrying in %.1fs", exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {url}")


def fetch_coin(symbol: str, days: int = HISTORY_DAYS) -> pd.DataFrame:
    pid = COINS[symbol]
    end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)
    step = timedelta(seconds=GRANULARITY * MAX_CANDLES)
    rows: list = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + step, end)
        rows += _get(URL.format(pid=pid), {"granularity": GRANULARITY, "start": cursor.isoformat(), "end": chunk_end.isoformat()})
        cursor = chunk_end
        time.sleep(0.35)  # stay well under public rate limits
    df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df["coin"] = symbol
    return df.drop_duplicates("time").sort_values("time").reset_index(drop=True)


def fetch_all(days: int = HISTORY_DAYS) -> pd.DataFrame:
    frames = []
    for sym in COINS:
        df = fetch_coin(sym, days)
        log.info("%s: %d candles", sym, len(df))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch_all()
    df.to_parquet(DATA_DIR / "candles.parquet", index=False)
    log.info("Saved %d rows", len(df))


if __name__ == "__main__":
    main()
