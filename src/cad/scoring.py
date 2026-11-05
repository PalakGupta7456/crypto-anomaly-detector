"""Anomaly scoring with the exported ONNX model (no PyTorch needed at serving time)."""
from __future__ import annotations

import json

import numpy as np
import onnxruntime as ort

from .config import FEATURES, MODEL_DIR

RECENT_STEPS = 3  # score focuses on the last hours of the window


def reconstruction_errors(x: np.ndarray, recon: np.ndarray) -> np.ndarray:
    return (x - recon) ** 2  # (n, window, features)


TOP_K = 3


def scores_from_errors(err: np.ndarray) -> np.ndarray:
    """Mean of the TOP_K largest errors among the last RECENT_STEPS hours x features.
    Averaging over everything would dilute a shock that hits a single feature."""
    recent = err[:, -RECENT_STEPS:, :].reshape(len(err), -1)
    return np.sort(recent, axis=1)[:, -TOP_K:].mean(axis=1)


def explain(err: np.ndarray) -> list[str]:
    """Which feature drove each score (largest error at the scored hour)."""
    return [FEATURES[i] for i in err[:, -1, :].argmax(axis=1)]


class Scorer:
    def __init__(self, model_dir=MODEL_DIR):
        self.session = ort.InferenceSession(str(model_dir / "autoencoder.onnx"), providers=["CPUExecutionProvider"])
        self.meta = json.loads((model_dir / "meta.json").read_text())

    def reconstruct(self, x: np.ndarray, batch: int = 4096) -> np.ndarray:
        outs = [self.session.run(None, {"x": x[i:i + batch]})[0] for i in range(0, len(x), batch)]
        return np.concatenate(outs) if outs else np.empty_like(x)

    def score(self, x: np.ndarray) -> tuple[np.ndarray, list[str]]:
        err = reconstruction_errors(x, self.reconstruct(x))
        return scores_from_errors(err), explain(err)

    @property
    def threshold(self) -> float:
        return float(self.meta["threshold"])
