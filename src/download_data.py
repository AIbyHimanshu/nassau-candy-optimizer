"""
download_data.py
----------------
Downloads the Nassau Candy dataset from Google Drive and saves it to data/nassau_raw.csv.
Run this first before any other script.
"""

import os
import pandas as pd

# ── Google Drive file ID (from the shared link) ────────────────────────────────
FILE_ID = "1c4VDb0Pf7RCgps4aLMiSuLtdaUpU_X49"
OUTPUT_PATH = "data/nassau_raw.csv"

# ── Factory coordinates (lat, lon) ─────────────────────────────────────────────
FACTORIES = {
    "Lot's O' Nuts":     (32.881893, -111.768036),
    "Wicked Choccy's":   (32.076176,  -81.088371),
    "Sugar Shack":       (48.11914,   -96.18115),
    "Secret Factory":    (41.446333,  -90.565487),
    "The Other Factory": (35.1175,    -89.971107),
}

# ── Product → Factory mapping ──────────────────────────────────────────────────
PRODUCT_FACTORY_MAP = {
    "Wonka Bar - Nutty Crunch Surprise":  "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows":          "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious":     "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate":         "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel":  "Wicked Choccy's",
    "Laffy Taffy":                        "Sugar Shack",
    "SweeTARTS":                          "Sugar Shack",
    "Nerds":                              "Sugar Shack",
    "Fun Dip":                            "Sugar Shack",
    "Fizzy Lifting Drinks":               "Sugar Shack",
    "Everlasting Gobstopper":             "Secret Factory",
    "Hair Toffee":                        "The Other Factory",
    "Lickable Wallpaper":                 "Secret Factory",
    "Wonka Gum":                          "Secret Factory",
    "Kazookles":                          "The Other Factory",
}

# ── Division mapping ───────────────────────────────────────────────────────────
PRODUCT_DIVISION_MAP = {
    "Wonka Bar - Nutty Crunch Surprise":  "Chocolate",
    "Wonka Bar - Fudge Mallows":          "Chocolate",
    "Wonka Bar -Scrumdiddlyumptious":     "Chocolate",
    "Wonka Bar - Milk Chocolate":         "Chocolate",
    "Wonka Bar - Triple Dazzle Caramel":  "Chocolate",
    "Laffy Taffy":                        "Sugar",
    "SweeTARTS":                          "Sugar",
    "Nerds":                              "Sugar",
    "Fun Dip":                            "Sugar",
    "Everlasting Gobstopper":             "Sugar",
    "Hair Toffee":                        "Sugar",
    "Fizzy Lifting Drinks":               "Other",
    "Lickable Wallpaper":                 "Other",
    "Wonka Gum":                          "Other",
    "Kazookles":                          "Other",
}


def download():
    os.makedirs("data", exist_ok=True)
    if os.path.exists(OUTPUT_PATH):
        print(f"[INFO] Data already exists at '{OUTPUT_PATH}'. Skipping download.")
        return

    print("[INFO] Downloading dataset from Google Drive...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    try:
        import gdown
        gdown.download(url, OUTPUT_PATH, quiet=False)
        print(f"[OK] Saved to '{OUTPUT_PATH}'")
    except Exception as e:
        print(f"[ERROR] gdown failed: {e}")
        print("[ACTION] Please download the file manually:")
        print(f"  URL: https://drive.google.com/file/d/{FILE_ID}/view")
        print(f"  Save as: {OUTPUT_PATH}")


if __name__ == "__main__":
    download()
    if os.path.exists(OUTPUT_PATH):
        df = pd.read_csv(OUTPUT_PATH)
        print(f"\n[INFO] Shape: {df.shape}")
        print(f"[INFO] Columns: {df.columns.tolist()}")
        print(df.head(3).to_string())
