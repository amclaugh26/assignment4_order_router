"""Router that picks the exchange with the best predicted price.
By Andrew McLaughlin
Last Updated: 11-28-2025

"""
#import libraries
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd

#set variables for use later
FEATURE_COLUMNS = ["side", "order_qty", "limit_price", "bid_price", "ask_price", "bid_size", "ask_size"]
MODEL_PATH = Path(__file__).resolve().parent / "models" / "order_router_models.joblib"

_MODELS: Dict[str, object] | None = None

#finds the models written to disk
def load_models(path: Path | str | None = None) -> Dict[str, object]:
    """Load or reuse the cached models dictionary."""
    global _MODELS
    if _MODELS is None:
        location = Path(path) if path else MODEL_PATH
        if not location.exists():
            raise FileNotFoundError(f"Models not found at {location}")
        _MODELS = joblib.load(location)
    return _MODELS

#allows for untrained models to be used in testing
def create_testing_model(models: Dict[str, object]) -> None:
    """Replace the cached models (used by tests to avoid disk I/O)."""
    global _MODELS
    _MODELS = models

#for a customer order, ensure it is the same format as the training data
def build_feature(
    side: str,
    quantity: int,
    limit_price: float,
    bid_price: float,
    ask_price: float,
    bid_size: int,
    ask_size: int,
) -> pd.DataFrame:
    """Put the incoming order into the same layout seen during training."""
    new_order = {
        "side": [side],
        "order_qty": [quantity],
        "limit_price": [limit_price],
        "bid_price": [bid_price],
        "ask_price": [ask_price],
        "bid_size": [bid_size],
        "ask_size": [ask_size],
    }
    return pd.DataFrame(new_order, columns=FEATURE_COLUMNS)

#return the best price improvement per exchange, sending back only the best exchange and price improvement
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
    """
    _ = symbol #ignore symbol
    models = load_models()
    features = build_feature(side.upper(), quantity, limit_price, bid_price, ask_price, bid_size, ask_size)
    #set initial values for best exchange and best_price_improvement to better handle weird inputs or results
    best_exchange: str | None = None
    best_price_improvement = float("-inf") 

    #run through the saved models and save the best exchanges
    for exchange, model in models.items():
        score = float(model.predict(features)[0])
        if score > best_price_improvement:
            best_price_improvement = score
            best_exchange = exchange

    #notify user if no models are present on disk
    if best_exchange is None:
        raise RuntimeError("No exchange models were loaded.")

    #returns the best exhange and price improvement
    return best_exchange, best_price_improvement
