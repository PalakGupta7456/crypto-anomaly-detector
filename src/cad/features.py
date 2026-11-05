"""Turn raw candles into scale-free features and sliding windows.

All features are robust z-scores (median / MAD over the previous week), so a
$60,000 coin and a $0.10 coin live on the same scale and one model can be
trained on every coin at once.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURES, NORM_WINDOW, WINDOW


def _robust_z(s: pd.Series, window: int = NORM_WINDOW) -> pd.Series:
    """z-score against the *previous* window only (shifted) so there is no look-ahead."""
    past = s.shift(1).rolling(window, min_periods=window // 2)
    med = past.median()
    mad = (s.shift(1) - med).abs().rolling(window, min_periods=window // 2).median() * 1.4826
    return ((s - med) / mad.replace(0, np.nan)).clip(-12, 12)


def coin_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("time").copy()
    ret = np.log(df["close"]).diff()
    logvol = np.log1p(df["volume"] * df["close"])  # dollar volume
    rng = (df["high"] - df["low"]) / df["close"]
    flow = ret * logvol  # signed volume-weighted move
    out = pd.DataFrame({
        "time": df["time"].values, "coin": df["coin"].values, "close": df["close"].values,
        "ret": ret.values,
        "ret_z": _robust_z(ret).values,
        "absret_z": _robust_z(ret.abs()).values,
        "vol_z": _robust_z(logvol).values,
        "range_z": _robust_z(rng).values,
        "flow_z": _robust_z(flow).values,
    })
    return out


def build_features(candles: pd.DataFrame) -> pd.DataFrame:
    feats = [coin_features(g) for _, g in candles.groupby("coin")]
    return pd.concat(feats, ignore_index=True).dropna(subset=FEATURES).reset_index(drop=True)


def make_windows(feat: pd.DataFrame, window: int = WINDOW) -> tuple[np.ndarray, pd.DataFrame]:
    """Sliding windows per coin. Returns X (n, window, n_features) and an index
    frame describing the *last* hour of each window (the hour being scored)."""
    xs, idx = [], []
    for _, g in feat.groupby("coin"):
        arr = g[FEATURES].to_numpy(np.float32)
        if len(arr) < window:
            continue
        view = np.lib.stride_tricks.sliding_window_view(arr, (window, arr.shape[1]))[:, 0]
        xs.append(view)
        idx.append(g.iloc[window - 1:][["time", "coin", "close", "ret"] + FEATURES].reset_index(drop=True))
    if not xs:
        return np.empty((0, window, len(FEATURES)), np.float32), pd.DataFrame()
    return np.concatenate(xs).astype(np.float32), pd.concat(idx, ignore_index=True)
