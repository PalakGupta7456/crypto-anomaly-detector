"""Score the latest market data and publish the dashboard JSON."""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from .config import COIN_NAMES, DATA_DIR, MODEL_DIR, SITE_DATA_DIR
from .features import build_features, make_windows
from .scoring import Scorer

log = logging.getLogger(__name__)
SHOW_HOURS = 24 * 7
REASONS = {
    "ret_z": "Unusual price move", "absret_z": "Unusual price move", "vol_z": "Volume burst",
    "range_z": "Volatility blow-out", "flow_z": "Abnormal buying or selling pressure",
}


def score_frame(candles: pd.DataFrame, scorer: Scorer) -> pd.DataFrame:
    x, idx = make_windows(build_features(candles))
    s, why = scorer.score(x)
    idx = idx.assign(score=s, driver=why)
    idx["severity"] = idx["score"] / scorer.threshold
    idx["anomaly"] = idx["severity"] > 1
    return idx


def update_event_log(scored: pd.DataFrame, path) -> pd.DataFrame:
    new = scored[scored["anomaly"]][["time", "coin", "score", "severity", "driver", "ret", "close"]]
    old = pd.read_csv(path, parse_dates=["time"]) if path.exists() else new.iloc[0:0]
    log_df = pd.concat([old, new]).drop_duplicates(["time", "coin"], keep="last").sort_values("time")
    log_df = log_df[log_df["time"] >= log_df["time"].max() - pd.Timedelta(days=90)] if len(log_df) else log_df
    log_df.to_csv(path, index=False)
    return log_df


def _r(v, n=4):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), n)


def build_payload(scored: pd.DataFrame, events: pd.DataFrame, meta: dict) -> dict:
    coins = []
    for sym, g in scored.groupby("coin", sort=False):
        g = g.sort_values("time").tail(SHOW_HOURS)
        last = g.iloc[-1]
        day = g[g["time"] > last["time"] - pd.Timedelta(hours=24)]
        coins.append({
            "symbol": sym, "name": COIN_NAMES.get(sym, sym),
            "price": _r(last["close"], 6), "change_24h": _r(day["close"].iloc[-1] / day["close"].iloc[0] - 1),
            "severity_now": _r(last["severity"], 3), "anomalies_7d": int(g["anomaly"].sum()),
            "series": {"time": [t.isoformat() for t in g["time"]], "close": [_r(v, 6) for v in g["close"]],
                       "severity": [_r(v, 3) for v in g["severity"]]},
            "heat": [_r(v, 2) for v in g["severity"].tail(48)],
        })
    coins.sort(key=lambda c: -(c["severity_now"] or 0))
    recent = events[events["time"] > events["time"].max() - pd.Timedelta(days=7)] if len(events) else events
    feed = [{
        "time": r.time.isoformat(), "coin": r.coin, "name": COIN_NAMES.get(r.coin, r.coin), "severity": _r(r.severity, 2),
        "reason": REASONS.get(r.driver, r.driver), "move": _r(r.ret, 4),
    } for r in recent.sort_values("time", ascending=False).head(40).itertuples()]
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "as_of": scored["time"].max().isoformat(),
        "model": {k: meta[k] for k in ("trained_at", "threshold", "benchmark", "at_threshold", "history", "n_train", "data_range")},
        "coins": coins, "feed": feed,
        "totals": {"events_7d": int(len(recent)), "events_logged": int(len(events))},
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    candles = pd.read_parquet(DATA_DIR / "candles.parquet")
    scorer = Scorer(MODEL_DIR)
    scored = score_frame(candles, scorer)
    SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    events = update_event_log(scored, SITE_DATA_DIR / "events.csv")
    payload = build_payload(scored, events, scorer.meta)
    (SITE_DATA_DIR / "latest.json").write_text(json.dumps(payload, separators=(",", ":")))
    log.info("Scored %d windows, %d anomalies in last 7 days", len(scored), payload["totals"]["events_7d"])


if __name__ == "__main__":
    main()
