"""Pytest file for the router
By Andrew McLaughlin
last-updated: 11-28-2025
"""
#import libraries
from __future__ import annotations

from pathlib import Path
import pytest
from joblib import load

import somewhat_smart_order_router as router

#create some stand-in models for ease of testing
class _ConstantModel:
    """Models that always returns a set value."""

    def __init__(self, price_improvement: float) -> None:
        self._price_improvement = price_improvement

    def predict(self, _: object) -> list[float]:
        return [self._price_improvement]


def test_best_price_improvement_prefers_largest_gain() -> None:
    """The router should choose the model with the max predicted improvement."""
    router.create_testing_model(
        {
            "EXCHANGE_A": _ConstantModel(0.2),
            "EXCHANGE_B": _ConstantModel(0.5),
        }
    )

    best_exchange, best_value = router.best_price_improvement(
        symbol="AAPL",
        side="B",
        quantity=100,
        limit_price=150.0,
        bid_price=149.5,
        ask_price=150.5,
        bid_size=200,
        ask_size=180,
    )

    assert best_exchange == "EXCHANGE_B"
    assert best_value == 0.5


def test_best_price_improvement_works_with_negative_forecasts() -> None:
    """Even if the predicted price improvement is negative, we pick the largest."""
    router.create_testing_model(
        {
            "EXCHANGE_LOW": _ConstantModel(-0.1),
            "EXCHANGE_HIGH": _ConstantModel(0.0),
        }
    )

    best_exchange, best_value = router.best_price_improvement(
        symbol="MSFT",
        side="S",
        quantity=50,
        limit_price=300.0,
        bid_price=299.0,
        ask_price=300.5,
        bid_size=50,
        ask_size=60,
    )

    assert best_exchange == "EXCHANGE_HIGH"
    assert best_value == 0.0


def test_best_price_improvement_prefers_first_on_tie() -> None:
    """When multiple exchanges tie, the router keeps the first seen."""
    router.create_testing_model(
        {
            "FIRST_EXCHANGE": _ConstantModel(0.3),
            "SECOND_EXCHANGE": _ConstantModel(0.3),
        }
    )

    best_exchange, best_value = router.best_price_improvement(
        symbol="TSLA",
        side="B",
        quantity=10,
        limit_price=700.0,
        bid_price=699.5,
        ask_price=700.1,
        bid_size=80,
        ask_size=60,
    )

    assert best_exchange == "FIRST_EXCHANGE"
    assert best_value == 0.3

def _load_serialized_models() -> dict[str, object]:
    path = Path(__file__).resolve().parent / "models" / "order_router_models.joblib"
    if not path.exists():
        pytest.skip("No models; train models first")
    return load(path)


def test_best_price_improvement_with_serialized_models() -> None:
    """The router still works when fed a loaded model."""
    models = _load_serialized_models()
    router.create_testing_model(models)

    exchange, score = router.best_price_improvement(
        symbol="AAPL",
        side="B",
        quantity=120,
        limit_price=150.0,
        bid_price=149.8,
        ask_price=150.1,
        bid_size=320,
        ask_size=310,
    )

    assert isinstance(exchange, str)
    assert isinstance(score, float)
    assert exchange in models


def test_best_price_improvement_serialized_models_second_order() -> None:
    """The router still works when fed a loaded model."""
    models = _load_serialized_models()
    router.create_testing_model(models)

    exchange, score = router.best_price_improvement(
        symbol="MSFT",
        side="S",
        quantity=90,
        limit_price=305.0,
        bid_price=304.6,
        ask_price=305.3,
        bid_size=210,
        ask_size=205,
    )

    assert isinstance(exchange, str)
    assert isinstance(score, float)
    assert exchange in models