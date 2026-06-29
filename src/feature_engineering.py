"""
feature_engineering.py
-----------------------
Encodes categorical features and assembles the feature matrix
used for ML model training and inference.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Categorical columns to label-encode
CAT_COLS = ["Ship Mode", "Factory", "Region", "Division"]

# Numeric columns used directly
NUM_COLS = ["Distance Miles", "Order Month", "Order Quarter", "Units", "Cost"]

# Target variable
TARGET = "Lead Time"


def build_features(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    """
    Build the feature matrix X and target vector y.

    Parameters
    ----------
    df       : processed DataFrame (output of preprocessing.py)
    encoders : dict of {col: LabelEncoder}. Pass existing encoders during
               inference to avoid refit (fit=False).
    fit      : if True, fit new encoders; if False, use supplied encoders.

    Returns
    -------
    X            : pd.DataFrame of feature columns
    y            : pd.Series of target values (Lead Time)
    encoders     : dict of fitted LabelEncoders
    feature_names: list of column names in X
    """
    df = df.copy()

    if encoders is None:
        encoders = {}

    for col in CAT_COLS:
        enc_col = col + "_enc"
        if fit:
            le = LabelEncoder()
            df[enc_col] = le.fit_transform(df[col].astype(str).fillna("Unknown"))
            encoders[col] = le
        else:
            le = encoders[col]
            # Handle unseen labels gracefully
            df[enc_col] = df[col].astype(str).fillna("Unknown").apply(
                lambda v: le.transform([v])[0] if v in le.classes_ else 0
            )

    enc_cols = [c + "_enc" for c in CAT_COLS]
    feature_names = enc_cols + NUM_COLS

    X = df[feature_names].fillna(0)
    y = df[TARGET] if TARGET in df.columns else pd.Series(dtype=float)

    return X, y, encoders, feature_names


def encode_single_row(encoders: dict, feature_names: list, **kwargs) -> pd.DataFrame:
    """
    Encode a single row for inference given raw (un-encoded) values.

    Example
    -------
    X = encode_single_row(
            encoders, feature_names,
            Ship_Mode="Standard Class",
            Factory="Sugar Shack",
            Region="South",
            Division="Sugar",
            Distance_Miles=850.0,
            Order_Month=6,
            Order_Quarter=2,
            Units=50,
            Cost=30
        )
    """
    # Map underscore keys to spaced column names
    key_map = {k.replace("_", " "): v for k, v in kwargs.items()}

    row = {}
    for col in CAT_COLS:
        le = encoders.get(col)
        val = str(key_map.get(col, "Unknown"))
        row[col + "_enc"] = le.transform([val])[0] if (le and val in le.classes_) else 0

    for col in NUM_COLS:
        row[col] = float(key_map.get(col, 0))

    return pd.DataFrame([row])[feature_names]
