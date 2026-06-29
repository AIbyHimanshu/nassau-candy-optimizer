"""
model_train.py
--------------
Trains three regression models to predict shipping lead time,
evaluates each on a held-out test set, selects the best by RMSE,
and saves the model + encoder artifacts to data/.

Run after preprocessing.py:
    python src/model_train.py
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, os.path.dirname(__file__))
from feature_engineering import build_features

PROCESSED_PATH  = "data/nassau_processed.csv"
MODEL_PATH      = "data/best_model.pkl"
ENCODERS_PATH   = "data/encoders.pkl"
FEATURES_PATH   = "data/feature_names.pkl"
METRICS_PATH    = "data/model_metrics.csv"
BEST_NAME_PATH  = "data/best_model_name.txt"


def evaluate(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    return {
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, preds))), 4),
        "MAE":  round(float(mean_absolute_error(y_test, preds)), 4),
        "R2":   round(float(r2_score(y_test, preds)), 4),
    }


def train_all(processed_path: str = PROCESSED_PATH) -> tuple:
    print(f"[INFO] Loading processed data from '{processed_path}'...")
    df = pd.read_csv(processed_path)

    X, y, encoders, feature_names = build_features(df, fit=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"[INFO] Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(
                                 n_estimators=200, random_state=42,
                                 n_jobs=-1, min_samples_leaf=5
                             ),
        "Gradient Boosting": GradientBoostingRegressor(
                                 n_estimators=200, learning_rate=0.1,
                                 max_depth=4, random_state=42
                             ),
    }

    results = {}
    print("\n── Model Training ──────────────────────────────────────")
    for name, model in candidates.items():
        print(f"  Training {name}...", end=" ", flush=True)
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        results[name] = {"model": model, **metrics}
        print(f"RMSE={metrics['RMSE']:.3f} | MAE={metrics['MAE']:.3f} | R²={metrics['R2']:.3f}")

    # Select best model by RMSE (lower is better)
    best_name = min(results, key=lambda k: results[k]["RMSE"])
    best_model = results[best_name]["model"]
    print(f"\n[OK] Best model: {best_name} (RMSE={results[best_name]['RMSE']:.3f})")

    # Save artifacts
    os.makedirs("data", exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump(encoders, f)
    with open(FEATURES_PATH, "wb") as f:
        pickle.dump(feature_names, f)
    with open(BEST_NAME_PATH, "w") as f:
        f.write(best_name)

    # Save metrics (without the model object)
    metrics_rows = {
        name: {k: v for k, v in vals.items() if k != "model"}
        for name, vals in results.items()
    }
    pd.DataFrame(metrics_rows).T.to_csv(METRICS_PATH)
    print(f"[OK] Artifacts saved to data/")

    return best_model, encoders, feature_names, results


if __name__ == "__main__":
    train_all()
