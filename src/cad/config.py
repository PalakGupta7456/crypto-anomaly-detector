"""Configuration for the crypto anomaly detector."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
SITE_DATA_DIR = ROOT / "docs" / "data"

# Coinbase Exchange product ids (public market data, no API key)
COINS = {
    "BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "XRP": "XRP-USD",
    "ADA": "ADA-USD", "DOGE": "DOGE-USD", "AVAX": "AVAX-USD", "LINK": "LINK-USD",
}
COIN_NAMES = {
    "BTC": "Bitcoin", "ETH": "Ethereum", "SOL": "Solana", "XRP": "XRP",
    "ADA": "Cardano", "DOGE": "Dogecoin", "AVAX": "Avalanche", "LINK": "Chainlink",
}

GRANULARITY = 3600          # 1-hour candles
HISTORY_DAYS = 90           # training history
WINDOW = 24                 # hours per sequence fed to the autoencoder
NORM_WINDOW = 168           # 1 week rolling window for robust z-scores
FEATURES = ["ret_z", "absret_z", "vol_z", "range_z", "flow_z"]
THRESHOLD_QUANTILE = 0.99  # anomaly when score exceeds this quantile of validation-period scores
SEED = 42
