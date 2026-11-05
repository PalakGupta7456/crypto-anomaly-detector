"""FastAPI service: score live crypto markets with the ONNX autoencoder.

Run:  uvicorn api.main:app --reload    ->  http://localhost:8000/docs
"""
from __future__ import annotations

import os
import time
import urllib.request
from contextlib import asynccontextmanager
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from cad import __version__
from cad.config import COIN_NAMES, COINS, MODEL_DIR
from cad.detect import REASONS, score_frame
from cad.ingest import fetch_coin
from cad.scoring import Scorer

STATE: dict = {}
CACHE_SECONDS = 10 * 60


def _ensure_model() -> None:
    base = os.getenv("MODEL_BASE_URL")  # e.g. raw URL of the live-data branch models/ folder
    for name in ("autoencoder.onnx", "meta.json"):
        path = MODEL_DIR / name
        if not path.exists() and base:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(f"{base.rstrip('/')}/{name}", path)
    if (MODEL_DIR / "autoencoder.onnx").exists():
        STATE["scorer"] = Scorer(MODEL_DIR)


@asynccontextmanager
async def lifespan(_: FastAPI):
    _ensure_model()
    yield


app = FastAPI(title="Crypto Anomaly Detector API", version=__version__, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])


@lru_cache(maxsize=16)
def _candles(symbol: str, bucket: int) -> pd.DataFrame:
    return fetch_coin(symbol, days=10)


def _scorer() -> Scorer:
    if "scorer" not in STATE:
        raise HTTPException(503, "Model not loaded")
    return STATE["scorer"]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "scorer" in STATE, "version": __version__}


@app.get("/coins")
def coins() -> list[dict]:
    return [{"symbol": s, "name": COIN_NAMES[s], "product": p} for s, p in COINS.items()]


@app.get("/model")
def model() -> dict:
    meta = _scorer().meta
    return {k: meta[k] for k in ("trained_at", "threshold", "benchmark", "at_threshold", "window", "features")}


@app.get("/score/{symbol}")
def score(symbol: str, hours: int = 24) -> dict:
    symbol = symbol.upper()
    if symbol not in COINS:
        raise HTTPException(404, f"Unknown coin '{symbol}'. See /coins")
    scored = score_frame(_candles(symbol, int(time.time() // CACHE_SECONDS)), _scorer()).tail(max(1, min(hours, 168)))
    return {
        "coin": symbol,
        "threshold": _scorer().threshold,
        "hours": [{"time": r.time.isoformat(), "close": float(r.close), "severity": round(float(r.severity), 3),
                   "anomaly": bool(r.anomaly), "reason": REASONS.get(r.driver, r.driver) if r.anomaly else None}
                  for r in scored.itertuples()],
    }
