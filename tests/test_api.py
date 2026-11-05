import pandas as pd
import pytest
from fastapi.testclient import TestClient

from cad.synthetic import generate


@pytest.fixture(scope="module")
def client():
    from api import main

    if not (main.MODEL_DIR / "autoencoder.onnx").exists():
        pytest.skip("No trained model")
    data = generate(days=12, end=pd.Timestamp.now().floor("h"))
    main._candles = lambda sym, bucket: data[data["coin"] == sym]  # no network in tests
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["model_loaded"] is True


def test_unknown_coin(client):
    assert client.get("/score/NOPE").status_code == 404


def test_score(client):
    body = client.get("/score/btc?hours=12").json()
    assert body["coin"] == "BTC" and len(body["hours"]) == 12
    assert all(h["severity"] >= 0 for h in body["hours"])
