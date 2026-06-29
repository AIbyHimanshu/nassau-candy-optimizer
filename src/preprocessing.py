"""
preprocessing.py
----------------
Loads the raw Nassau Candy dataset, computes shipping distances, engineers
realistic lead times from distance + ship mode, computes profit margins,
and produces a clean feature-rich DataFrame saved to data/nassau_processed.csv.

Run after download_data.py:
    python src/preprocessing.py
"""

import os
import sys
import pandas as pd
import numpy as np
from geopy.distance import geodesic

sys.path.insert(0, os.path.dirname(__file__))
from download_data import FACTORIES, PRODUCT_FACTORY_MAP, PRODUCT_DIVISION_MAP

# ── US State centroids (lat, lon) ──────────────────────────────────────────────
STATE_CENTROIDS = {
    "Alabama":        (32.806671,  -86.791130),
    "Alaska":         (61.370716, -152.404419),
    "Arizona":        (33.729759, -111.431221),
    "Arkansas":       (34.969704,  -92.373123),
    "California":     (36.116203, -119.681564),
    "Colorado":       (39.059811, -105.311104),
    "Connecticut":    (41.597782,  -72.755371),
    "Delaware":       (39.318523,  -75.507141),
    "Florida":        (27.766279,  -81.686783),
    "Georgia":        (33.040619,  -83.643074),
    "Hawaii":         (21.094318, -157.498337),
    "Idaho":          (44.240459, -114.478828),
    "Illinois":       (40.349457,  -88.986137),
    "Indiana":        (39.849426,  -86.258278),
    "Iowa":           (42.011539,  -93.210526),
    "Kansas":         (38.526600,  -96.726486),
    "Kentucky":       (37.668140,  -84.670067),
    "Louisiana":      (31.169960,  -91.867805),
    "Maine":          (44.693947,  -69.381927),
    "Maryland":       (39.063946,  -76.802101),
    "Massachusetts":  (42.230171,  -71.530106),
    "Michigan":       (43.326618,  -84.536095),
    "Minnesota":      (45.694454,  -93.900192),
    "Mississippi":    (32.741646,  -89.678696),
    "Missouri":       (38.456085,  -92.288368),
    "Montana":        (46.921925, -110.454353),
    "Nebraska":       (41.125370,  -98.268082),
    "Nevada":         (38.313515, -117.055374),
    "New Hampshire":  (43.452492,  -71.563896),
    "New Jersey":     (40.298904,  -74.521011),
    "New Mexico":     (34.840515, -106.248482),
    "New York":       (42.165726,  -74.948051),
    "North Carolina": (35.630066,  -79.806419),
    "North Dakota":   (47.528912,  -99.784012),
    "Ohio":           (40.388783,  -82.764915),
    "Oklahoma":       (35.565342,  -96.928917),
    "Oregon":         (44.572021, -122.070938),
    "Pennsylvania":   (40.590752,  -77.209755),
    "Rhode Island":   (41.680893,  -71.511780),
    "South Carolina": (33.856892,  -80.945007),
    "South Dakota":   (44.299782,  -99.438828),
    "Tennessee":      (35.747845,  -86.692345),
    "Texas":          (31.054487,  -97.563461),
    "Utah":           (40.150032, -111.862434),
    "Vermont":        (44.045876,  -72.710686),
    "Virginia":       (37.769337,  -78.169968),
    "Washington":     (47.400902, -121.490494),
    "West Virginia":  (38.491226,  -80.954453),
    "Wisconsin":      (44.268543,  -89.616508),
    "Wyoming":        (42.755966, -107.302490),
    "New York City":  (40.712776,  -74.005974),
    "District of Columbia": (38.907192, -77.036871),
}

US_CENTER = (39.5, -98.35)

# ── Ship mode base days (industry-standard estimates) ──────────────────────────
SHIP_MODE_BASE = {
    "Same Day":       1.0,
    "First Class":    2.5,
    "Second Class":   4.5,
    "Standard Class": 7.0,
}
DEFAULT_BASE   = 5.0
MILES_PER_DAY  = 500.0   # ~500 miles processed per day in ground shipping


def _get_dest_coords(row: pd.Series) -> tuple:
    """Return (lat, lon) for a row's destination: city → state → US centre."""
    city  = str(row.get("City", "")).strip()
    state = str(row.get("State/Province", "")).strip()
    if city in STATE_CENTROIDS:
        return STATE_CENTROIDS[city]
    if state in STATE_CENTROIDS:
        return STATE_CENTROIDS[state]
    return US_CENTER


def _compute_distance(factory_name: str, dest_coords: tuple) -> float:
    if factory_name not in FACTORIES:
        return np.nan
    return geodesic(FACTORIES[factory_name], dest_coords).miles


def load_and_preprocess(path: str = "data/nassau_raw.csv") -> pd.DataFrame:
    print(f"[INFO] Loading raw data from '{path}'...")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    print(f"[INFO] Raw shape: {df.shape}")

    # ── Date parsing (no lead time from dates — ship dates are placeholder) ────
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
    df["Ship Date"]  = pd.to_datetime(df["Ship Date"],  dayfirst=True, errors="coerce")
    df.dropna(subset=["Order Date", "Ship Date"], inplace=True)

    df["Order Month"]   = df["Order Date"].dt.month
    df["Order Quarter"] = df["Order Date"].dt.quarter
    df["Order Year"]    = df["Order Date"].dt.year

    # ── Factory & division mapping ─────────────────────────────────────────────
    df["Factory"]  = df["Product Name"].map(PRODUCT_FACTORY_MAP)
    df["Division"] = df["Product Name"].map(PRODUCT_DIVISION_MAP)

    before = len(df)
    df.dropna(subset=["Factory"], inplace=True)
    print(f"[INFO] Dropped {before - len(df)} rows with unmapped products.")

    # ── Shipping distance ──────────────────────────────────────────────────────
    print("[INFO] Computing shipping distances (this may take a moment)...")
    dest_coords_list = df.apply(_get_dest_coords, axis=1)
    df["Distance Miles"] = [
        _compute_distance(fac, coords)
        for fac, coords in zip(df["Factory"], dest_coords_list)
    ]

    # ── Engineer realistic lead time ───────────────────────────────────────────
    # Lead time = ship-mode base days + distance effect + small random noise.
    # The raw ship dates in this dataset are placeholder values (all ~2026),
    # so we derive lead time from the operational factors that actually drive it.
    np.random.seed(42)
    ship_base       = df["Ship Mode"].map(SHIP_MODE_BASE).fillna(DEFAULT_BASE)
    distance_effect = df["Distance Miles"] / MILES_PER_DAY
    noise           = np.random.normal(0, 0.4, size=len(df))
    df["Lead Time"] = (ship_base + distance_effect + noise).clip(lower=1.0).round(1)

    # ── Financial metrics ──────────────────────────────────────────────────────
    df["Sales"]        = pd.to_numeric(df.get("Sales",        0), errors="coerce").fillna(0)
    df["Gross Profit"] = pd.to_numeric(df.get("Gross Profit", 0), errors="coerce").fillna(0)
    df["Cost"]         = pd.to_numeric(df.get("Cost",         0), errors="coerce").fillna(0)
    df["Units"]        = pd.to_numeric(df.get("Units",        1), errors="coerce").fillna(1)

    df["Profit Margin"] = np.where(
        df["Sales"] > 0,
        df["Gross Profit"] / df["Sales"],
        np.nan
    )

    # ── Remove statistical outliers in engineered lead time (IQR) ─────────────
    Q1, Q3 = df["Lead Time"].quantile([0.25, 0.75])
    IQR    = Q3 - Q1
    lower  = max(Q1 - 3 * IQR, 1.0)
    upper  = Q3 + 3 * IQR
    df = df[(df["Lead Time"] >= lower) & (df["Lead Time"] <= upper)].copy()

    print(f"[OK] Processed shape: {df.shape}")
    print(f"[INFO] Lead Time range: {df['Lead Time'].min():.1f} – {df['Lead Time'].max():.1f} days "
          f"(mean: {df['Lead Time'].mean():.1f})")
    return df


if __name__ == "__main__":
    df = load_and_preprocess()
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/nassau_processed.csv", index=False)
    print(f"[OK] Saved to 'data/nassau_processed.csv'")
    print("\nSample:")
    print(df[["Product Name", "Factory", "Ship Mode", "Lead Time",
              "Distance Miles", "Profit Margin", "Order Month"]].head(8).to_string())