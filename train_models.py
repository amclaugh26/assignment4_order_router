"""Train per-exchange price improvement models and persist them for inference."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm.auto import tqdm

from data_processing import prepare_training_data

FEATURE_COLUMNS = ["side", "order_qty", "limit_price", "bid_price", "ask_price", "bid_size", "ask_size"]
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "order_router_models.joblib"


def _build_onehot() -> OneHotEncoder:
    """Return a OneHotEncoder that works across sklearn versions."""
    try:
        return OneHotEncoder(drop="if_binary", sparse_output=False)
    except TypeError:  # some sklearn versions still expect `sparse`
        return OneHotEncoder(drop="if_binary", sparse=False)


def build_pipeline() -> Pipeline:
    """Create the preprocessing -> regressor pipeline for each exchange."""
    numeric_features = [
        "order_qty",
        "limit_price",
        "bid_price",
        "ask_price",
        "bid_size",
        "ask_size",
    ]
    preprocessor = ColumnTransformer(
        transformers=[
            ("side_encoder", _build_onehot(), ["side"]),
            ("numeric_scaler", StandardScaler(), numeric_features),
        ],
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(random_state=42, n_jobs=-1)),
        ]
    )
    return pipeline


def train_models(df: pd.DataFrame) -> Dict[str, Pipeline]:
    """Fit a tuned model for each exchange in the data."""
    pipelines: Dict[str, Pipeline] = {}
    grouped = list(df.groupby("exchange"))
    for exchange, group in tqdm(grouped, desc="training exchanges"):
        if group.shape[0] < 10:
            # Skip exchanges with too little data for reliable training.
            continue
        X = group[FEATURE_COLUMNS]
        y = group["price_improvement"]
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )

        pipeline = build_pipeline()
        param_grid = {
            "regressor__n_estimators": [80, 120],
            "regressor__max_depth": [None, 8, 12],
            "regressor__min_samples_split": [2, 5],
        }
        grid = GridSearchCV(
            pipeline,
            param_grid,
            scoring="neg_root_mean_squared_error",
            cv=3,
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"{exchange}: RMSE={rmse:.4f} R2={r2:.4f} (best params: {grid.best_params_})")
        pipelines[exchange] = best_model
    return pipelines


def persist_models(models: Mapping[str, Pipeline], path: Path) -> None:
    """Write the dictionary of trained models to disk with joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, path)
    print(f"Saved {len(models)} exchange models to {path}")


def run_training(
    *,
    executions_path: Path | str = Path("/opt/assignment3/execs_from_fix.csv"),
    quotes_path: Path | str = Path("/opt/assignment4/quotes_2025-09-10_small.csv.gz"),
    model_path: Path = MODEL_PATH,
    max_symbols: int | None = 25,
) -> Path:
    """Prepare the data, train models, and save them for inference."""
    df = prepare_training_data(
        executions_path=executions_path,
        quotes_path=quotes_path,
        max_symbols=max_symbols,
    )
    models = train_models(df)
    if not models:
        raise RuntimeError("No exchange received enough data to train.")

    persist_models(models, model_path)
    return model_path


def main() -> None:
    """Entry point for CLI execution."""
    run_training(max_symbols=None)


if __name__ == "__main__":
    main()
