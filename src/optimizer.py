"""
optimizer.py
------------
Core simulation and recommendation engine.

Key functions
-------------
load_artifacts()                   → loads model, encoders, feature_names from data/
simulate_factory_reassignment(...) → predicts lead time for a product across every factory
generate_recommendations(...)      → ranks best reassignment moves across all products
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from geopy.distance import geodesic

sys.path.insert(0, os.path.dirname(__file__))
from download_data import FACTORIES, PRODUCT_FACTORY_MAP
from preprocessing import STATE_CENTROIDS

# Fallback US centre
US_CENTER = (39.5, -98.35)


# ── Artifact loading ───────────────────────────────────────────────────────────

def load_artifacts():
    """Load model, encoders, and feature names from disk."""
    with open("data/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("data/encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    with open("data/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    return model, encoders, feature_names


def _safe_encode(encoders: dict, col: str, val: str) -> int:
    le = encoders.get(col)
    if le is None:
        return 0
    val = str(val)
    return int(le.transform([val])[0]) if val in le.classes_ else 0


# ── Single-row prediction ──────────────────────────────────────────────────────

def predict_lead_time(
    model, encoders, feature_names,
    factory, ship_mode, region, division,
    distance, month, quarter, units, cost
) -> float:
    """Predict lead time (days) for a single scenario."""
    row = {
        "Ship Mode_enc":  _safe_encode(encoders, "Ship Mode", ship_mode),
        "Factory_enc":    _safe_encode(encoders, "Factory",   factory),
        "Region_enc":     _safe_encode(encoders, "Region",    region),
        "Division_enc":   _safe_encode(encoders, "Division",  division),
        "Distance Miles": float(distance),
        "Order Month":    int(month),
        "Order Quarter":  int(quarter),
        "Units":          float(units),
        "Cost":           float(cost),
    }
    X = pd.DataFrame([row])[feature_names]
    return float(model.predict(X)[0])


# ── Factory simulation ─────────────────────────────────────────────────────────

def simulate_factory_reassignment(
    product_name: str,
    dest_state: str,
    ship_mode: str,
    region: str,
    division: str,
    month: int,
    quarter: int,
    units: float,
    cost: float,
    model,
    encoders: dict,
    feature_names: list,
) -> tuple:
    """
    For a given product and destination, predict lead time if produced at
    every available factory. Returns a ranked DataFrame and the current factory.
    """
    current_factory = PRODUCT_FACTORY_MAP.get(product_name, list(FACTORIES.keys())[0])
    dest_coords = STATE_CENTROIDS.get(dest_state, US_CENTER)

    rows = []
    for factory_name, factory_coords in FACTORIES.items():
        distance = geodesic(factory_coords, dest_coords).miles
        lt = predict_lead_time(
            model, encoders, feature_names,
            factory_name, ship_mode, region, division,
            distance, month, quarter, units, cost
        )
        rows.append({
            "Factory":              factory_name,
            "Distance Miles":       round(distance, 1),
            "Predicted Lead Time":  round(max(lt, 0), 2),
            "Is Current":           factory_name == current_factory,
        })

    df = pd.DataFrame(rows).sort_values("Predicted Lead Time").reset_index(drop=True)

    current_lt = df.loc[df["Is Current"], "Predicted Lead Time"].values
    current_lt = float(current_lt[0]) if len(current_lt) else df["Predicted Lead Time"].max()

    df["Lead Time Reduction (days)"] = round(current_lt - df["Predicted Lead Time"], 2)
    df["Lead Time Reduction (%)"]    = round(
        (current_lt - df["Predicted Lead Time"]) / max(current_lt, 0.01) * 100, 1
    )
    # Profit impact: penalise longer distances (max distance = 30 % cost uplift)
    max_dist = df["Distance Miles"].max() or 1
    df["Profit Impact Score"] = round(
        1 - (df["Distance Miles"] / max_dist) * 0.30, 3
    )

    return df, current_factory


# ── Batch recommendation engine ────────────────────────────────────────────────

def generate_recommendations(
    df_processed: pd.DataFrame,
    model,
    encoders: dict,
    feature_names: list,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Iterate over every product, simulate all factory alternatives, and return
    the top-N moves ranked by lead-time reduction.
    """
    recs = []

    for product, current_factory in PRODUCT_FACTORY_MAP.items():
        prod_df = df_processed[df_processed["Product Name"] == product]
        if prod_df.empty:
            continue

        # Use modal / representative values for this product
        sample = prod_df.iloc[0]
        state     = str(sample.get("State/Province", "Texas")).strip()
        ship_mode = str(sample.get("Ship Mode", "Standard Class")).strip()
        region    = str(sample.get("Region", "South")).strip()
        division  = str(sample.get("Division", "Sugar")).strip()
        month     = int(sample.get("Order Month", 6))
        quarter   = int(sample.get("Order Quarter", 2))
        units     = float(sample.get("Units", 10))
        cost      = float(sample.get("Cost", 50))

        dest_coords    = STATE_CENTROIDS.get(state, US_CENTER)
        current_coords = FACTORIES[current_factory]
        current_dist   = geodesic(current_coords, dest_coords).miles
        current_lt     = predict_lead_time(
            model, encoders, feature_names,
            current_factory, ship_mode, region, division,
            current_dist, month, quarter, units, cost
        )

        for alt_factory, alt_coords in FACTORIES.items():
            if alt_factory == current_factory:
                continue
            alt_dist = geodesic(alt_coords, dest_coords).miles
            alt_lt   = predict_lead_time(
                model, encoders, feature_names,
                alt_factory, ship_mode, region, division,
                alt_dist, month, quarter, units, cost
            )
            reduction = current_lt - alt_lt
            if reduction <= 0:
                continue

            recs.append({
                "Product":                    product,
                "Current Factory":            current_factory,
                "Recommended Factory":        alt_factory,
                "Current Lead Time (days)":   round(current_lt, 2),
                "Predicted Lead Time (days)": round(alt_lt, 2),
                "Lead Time Reduction (days)": round(reduction, 2),
                "Lead Time Reduction (%)":    round(reduction / max(current_lt, 0.01) * 100, 1),
                "Distance Delta (miles)":     round(alt_dist - current_dist, 1),
                "Profit Impact Score":        round(
                    1 - (alt_dist / max(alt_dist, current_dist)) * 0.30, 3
                ),
            })

    if not recs:
        return pd.DataFrame()

    return (
        pd.DataFrame(recs)
        .sort_values("Lead Time Reduction (days)", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# ── Risk classification ────────────────────────────────────────────────────────

def classify_risk(row: pd.Series) -> str:
    """Classify a recommendation row as High / Medium / Low risk."""
    if row["Distance Delta (miles)"] > 1000:
        return "High"
    if row["Lead Time Reduction (%)"] < 5:
        return "Low"
    return "Medium"


def add_risk_labels(recs_df: pd.DataFrame) -> pd.DataFrame:
    if recs_df.empty:
        return recs_df
    recs_df = recs_df.copy()
    recs_df["Risk Level"] = recs_df.apply(classify_risk, axis=1)
    return recs_df
