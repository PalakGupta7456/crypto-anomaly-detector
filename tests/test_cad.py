import numpy as np
import pandas as pd
import pytest

from cad.config import COINS, FEATURES, WINDOW
from cad.features import build_features, coin_features, make_windows
from cad.scoring import scores_from_errors
from cad.synthetic import generate
from cad.train import inject_anomalies

END = pd.Timestamp("2026-01-31 00:00")


@pytest.fixture(scope="module")
def candles():
    return generate(days=30, end=END)


def test_synthetic_is_sane(candles):
    assert set(candles["coin"]) == set(COINS)
    assert np.isfinite(candles[["open", "high", "low", "close", "volume"]].to_numpy()).all()
    assert (candles["high"] >= candles["low"]).all()


def test_zscores_have_no_lookahead(candles):
    """Changing a future price must not change any earlier feature value."""
    g = candles[candles["coin"] == "BTC"].reset_index(drop=True)
    base = coin_features(g)
    g2 = g.copy()
    g2.loc[len(g2) - 1, "close"] *= 3
    moved = coin_features(g2)
    pd.testing.assert_frame_equal(base.iloc[:-1], moved.iloc[:-1])


def test_windows_shape(candles):
    x, idx = make_windows(build_features(candles))
    assert x.shape[1:] == (WINDOW, len(FEATURES))
    assert len(x) == len(idx)
    assert x.dtype == np.float32


def test_injection_labels(candles):
    x, _ = make_windows(build_features(candles))
    xi, y, kinds = inject_anomalies(x, rate=0.05)
    assert y.sum() == int(len(x) * 0.05) == len(kinds)
    changed = np.any(xi != x, axis=(1, 2))
    assert (changed == y.astype(bool)).all()


def test_score_prefers_concentrated_error():
    err = np.zeros((2, WINDOW, len(FEATURES)), np.float32)
    err[0, -1, 0] = 9.0                  # one big error at the scored hour
    err[1, :, :] = 9.0 / (WINDOW * len(FEATURES))  # same total error, spread out
    s = scores_from_errors(err)
    assert s[0] > s[1]


def test_train_export_and_score(candles, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import cad.train as tr
    from cad.detect import build_payload, score_frame, update_event_log
    from cad.scoring import Scorer

    monkeypatch.setattr(tr, "MODEL_DIR", tmp_path)
    meta = tr.train(candles, epochs=2)
    assert {"lstm_autoencoder", "isolation_forest", "zscore_rule"} <= set(meta["benchmark"])
    scorer = Scorer(tmp_path)
    scored = score_frame(candles, scorer)
    assert scored["severity"].notna().all()
    events = update_event_log(scored, tmp_path / "events.csv")
    payload = build_payload(scored, events, scorer.meta)
    assert len(payload["coins"]) == len(COINS)
