"""Inference utilities that pick the exchange with the best predicted price."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd

FEATURE_COLUMNS = ["side", "order_qty", "limit_price", "bid_price", "ask_price", "bid_size", "ask_size"]
MODEL_PATH = Path(__file__).resolve().parent / "models" / "order_router_models.joblib"

_MODELS: Dict[str, object] | None = None


def _load_models(path: Path | str | None = None) -> Dict[str, object]:
    """Load or reuse the cached models dictionary."""
    global _MODELS
    if _MODELS is None:
        location = Path(path) if path else MODEL_PATH
        if not location.exists():
            raise FileNotFoundError(f"Models not found at {location}")
        _MODELS = joblib.load(location)
    return _MODELS


def register_models_for_test(models: Dict[str, object]) -> None:
    """Replace the cached models (used by tests to avoid disk I/O)."""
    global _MODELS
    _MODELS = models


def _build_feature_frame(
    side: str,
    quantity: int,
    limit_price: float,
    bid_price: float,
    ask_price: float,
    bid_size: int,
    ask_size: int,
) -> pd.DataFrame:
    """Put the incoming order into the same layout seen during training."""
    payload = {
        "side": [side],
        "order_qty": [quantity],
        "limit_price": [limit_price],
        "bid_price": [bid_price],
        "ask_price": [ask_price],
        "bid_size": [bid_size],
        "ask_size": [ask_size],
    }
    return pd.DataFrame(payload, columns=FEATURE_COLUMNS)


def best_price_improvement(
    symbol: str,
    side: str,
    quantity: int,
    limit_price: float,
    bid_price: float,
    ask_price: float,
    bid_size: int,
    ask_size: int,
) -> Tuple[str, float]:
    """
    Return the exchange name with the highest predicted price improvement.

    The function ignores `symbol` because the models are exchange-specific.
    """
    _ = symbol
    models = _load_models()
    features = _build_feature_frame(side.upper(), quantity, limit_price, bid_price, ask_price, bid_size, ask_size)
    best_exchange: str | None = None
    best_score = float("-inf")

    for exchange, model in models.items():
        score = float(model.predict(features)[0])
        if score > best_score:
            best_score = score
            best_exchange = exchange

    if best_exchange is None:
        raise RuntimeError("No exchange models were loaded.")

    return best_exchange, best_score
