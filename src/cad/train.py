"""Train the LSTM autoencoder, benchmark it, export to ONNX.

Real market anomalies are unlabeled, so evaluation uses the standard
*synthetic injection* protocol: known shocks (price spikes, volume bursts
without price moves, volatility blow-outs, pump-and-dump shapes) are injected
into held-out data, and each detector is scored on how well it ranks them.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import FEATURES, MODEL_DIR, SEED, THRESHOLD_QUANTILE, WINDOW
from .features import build_features, make_windows
from .scoring import reconstruction_errors, scores_from_errors

log = logging.getLogger(__name__)
F = {f: i for i, f in enumerate(FEATURES)}


def inject_anomalies(x: np.ndarray, rate: float = 0.02, seed: int = SEED) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    x = x.copy()
    y = np.zeros(len(x), dtype=int)
    kinds = []
    for i in rng.choice(len(x), int(len(x) * rate), replace=False):
        kind = rng.choice(["price_spike", "volume_burst", "volatility", "pump_dump"])
        s = rng.choice([-1, 1])
        if kind == "price_spike":
            x[i, -1, F["ret_z"]] += s * rng.uniform(5, 9)
            x[i, -1, F["absret_z"]] += rng.uniform(5, 9)
        elif kind == "volume_burst":           # big volume, price flat: possible wash trading
            x[i, -1, F["vol_z"]] += rng.uniform(4, 8)
            x[i, -1, F["flow_z"]] += rng.uniform(1, 3)
        elif kind == "volatility":
            x[i, -2:, F["range_z"]] += rng.uniform(4, 8)
            x[i, -2:, F["absret_z"]] += rng.uniform(2, 4)
        else:                                  # pump then dump within 3 hours
            x[i, -3, F["ret_z"]] += rng.uniform(4, 7)
            x[i, -1, F["ret_z"]] -= rng.uniform(4, 7)
            x[i, -3:, F["vol_z"]] += rng.uniform(2, 5)
        y[i] = 1
        kinds.append(str(kind))
    return x, y, kinds


def _eval(y, s) -> dict:
    return {"roc_auc": round(float(roc_auc_score(y, s)), 4), "pr_auc": round(float(average_precision_score(y, s)), 4)}


def train(candles: pd.DataFrame, epochs: int = 25) -> dict:
    import torch  # noqa: F401  (imported lazily: serving does not need torch)

    from .model import export_onnx, train_autoencoder

    feat = build_features(candles)
    x, idx = make_windows(feat)
    order = np.argsort(idx["time"].values, kind="stable")
    x, idx = x[order], idx.iloc[order].reset_index(drop=True)
    n = len(x)
    tr_end, va_end = int(n * 0.70), int(n * 0.85)
    x_tr, x_va, x_te = x[:tr_end], x[tr_end:va_end], x[va_end:]
    log.info("windows: train %d  val %d  test %d", len(x_tr), len(x_va), len(x_te))

    model, history = train_autoencoder(x_tr, x_va, epochs=epochs, log=log.info)

    def ae_scores(arr):
        with torch.no_grad():
            recon = np.concatenate([model(torch.from_numpy(arr[i:i + 4096])).numpy() for i in range(0, len(arr), 4096)])
        return scores_from_errors(reconstruction_errors(arr, recon))

    # Threshold from the validation period (unseen by training), so it reflects out-of-sample error levels
    threshold = float(np.quantile(ae_scores(x_va), THRESHOLD_QUANTILE))

    # Benchmark on held-out data with injected anomalies
    x_inj, y_inj, _ = inject_anomalies(x_te)
    s_ae = ae_scores(x_inj)
    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=SEED).fit(x_tr.reshape(len(x_tr), -1))
    s_iso = -iso.score_samples(x_inj.reshape(len(x_inj), -1))
    s_rule = np.abs(x_inj[:, -1, F["ret_z"]])  # "flag big moves" rule of thumb
    flagged = s_ae > threshold
    tp = int((flagged & (y_inj == 1)).sum())
    bench = {
        "lstm_autoencoder": _eval(y_inj, s_ae),
        "isolation_forest": _eval(y_inj, s_iso),
        "zscore_rule": _eval(y_inj, s_rule),
    }
    precision = tp / max(int(flagged.sum()), 1)
    recall = tp / max(int(y_inj.sum()), 1)
    k = int(y_inj.sum())
    for name, s in (("lstm_autoencoder", s_ae), ("isolation_forest", s_iso), ("zscore_rule", s_rule)):
        bench[name]["precision_at_k"] = round(float(y_inj[np.argsort(-s)[:k]].mean()), 4)
    log.info("benchmark: %s | precision %.3f recall %.3f", bench, precision, recall)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    export_onnx(model, WINDOW, len(FEATURES), MODEL_DIR / "autoencoder.onnx")
    meta = {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "threshold": threshold, "threshold_quantile": THRESHOLD_QUANTILE,
        "window": WINDOW, "features": FEATURES,
        "n_train": int(len(x_tr)), "n_val": int(len(x_va)), "n_test": int(len(x_te)),
        "benchmark": bench, "at_threshold": {"precision": round(precision, 4), "recall": round(recall, 4)},
        "history": history,
        "data_range": [str(idx["time"].min()), str(idx["time"].max())],
    }
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    from .config import DATA_DIR
    logging.basicConfig(level=logging.INFO)
    train(pd.read_parquet(DATA_DIR / "candles.parquet"))


if __name__ == "__main__":
    main()
