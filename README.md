# 🍬 Nassau Candy Distributor — Factory Reallocation & Shipping Optimization System

exit[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://candy-factory.streamlit.app/)

A machine learning-powered **decision intelligence system** that recommends optimal factory assignments for Nassau Candy products — minimising shipping lead times and protecting profit margins across 5 factories and 15 products.

> **Live Dashboard →** [candy-factory.streamlit.app](https://candy-factory.streamlit.app/)

---

## Project Overview

Nassau Candy currently assigns products to factories using static rules and legacy processes, leading to suboptimal shipping distances, high lead times for certain regions, and margin erosion due to logistics inefficiencies.

This system elevates Nassau Candy from descriptive analytics to **intelligent decision-making** by:

- Predicting shipping lead times under different factory-product configurations
- Recommending which products should be reassigned to alternative factories
- Simulating what-if scenarios before any operational change is executed
- Balancing shipping efficiency against profitability in every recommendation

---

## Dataset

| Field | Description |
|---|---|
| 10,194 orders | Jan 2024 across all factories |
| 15 products | Across 3 divisions (Chocolate, Sugar, Other) |
| 5 factories | Across the continental US |
| 4 ship modes | Same Day, First Class, Second Class, Standard Class |
| 18 raw fields | Orders, dates, customers, products, sales, costs, profit |

---

## Factories & Products

| Factory | Location | Products |
|---|---|---|
| Lot's O' Nuts | Arizona | Wonka Bar - Nutty Crunch Surprise, Fudge Mallows, Scrumdiddlyumptious |
| Wicked Choccy's | Georgia | Wonka Bar - Milk Chocolate, Triple Dazzle Caramel |
| Sugar Shack | Minnesota | Laffy Taffy, SweeTARTS, Nerds, Fun Dip, Fizzy Lifting Drinks |
| Secret Factory | Illinois | Everlasting Gobstopper, Lickable Wallpaper, Wonka Gum |
| The Other Factory | Tennessee | Hair Toffee, Kazookles |

---

## ML Models & Results

Three regression models were trained to predict shipping lead time given product, factory, destination region, ship mode, and distance:

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | 0.820 days | 0.594 days | 0.887 |
| Random Forest | 0.420 days | 0.333 days | 0.970 |
| **Gradient Boosting** | **0.410 days** | **0.325 days** | **0.972** |

**Best model: Gradient Boosting** — R² of 0.972 means the model explains 97.2% of lead time variance with predictions accurate to within ±0.4 days on average.

---

## Dashboard Pages

| Page | Description |
|---|---|
| **Home** | Pipeline status, model performance summary |
| **Factory Optimizer** | Select any product + destination → see predicted lead time across all 5 factories |
| **What-If Analysis** | Compare current vs proposed factory assignment with waterfall chart and gauges |
| **Recommendations** | Ranked factory reassignment suggestions with risk labels and CSV export |
| **Risk Panel** | Profit sensitivity, high-risk alerts, lead time trends, model performance charts |

---

## Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/nassau-candy-optimizer.git
cd nassau-candy-optimizer
```

### 2. Create and activate virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
python -m pip install --no-user -r requirements.txt
```

### 4. Run the data pipeline
```bash
python src/download_data.py    # Downloads raw dataset from Google Drive
python src/preprocessing.py   # Engineers features + distances + lead times
python src/model_train.py      # Trains 3 models, saves best to data/
```

### 5. Launch the dashboard
```bash
streamlit run streamlit_app/app.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## Project Structure

```
nassau-candy-optimizer/
├── data/                          # Generated artifacts (CSV + model .pkl files)
├── notebooks/
│   └── 01_eda.ipynb               # Annotated exploratory data analysis notebook
├── src/
│   ├── download_data.py           # Google Drive download + factory/product maps
│   ├── preprocessing.py           # Feature engineering, distances, lead times
│   ├── feature_engineering.py     # Encodes features for ML
│   ├── model_train.py             # Trains 3 models, evaluates, saves best
│   └── optimizer.py               # Simulation engine + recommendation logic
├── streamlit_app/
│   ├── app.py                     # Landing page
│   └── pages/
│       ├── 1_Factory_Optimizer.py
│       ├── 2_WhatIf_Analysis.py
│       ├── 3_Recommendations.py
│       └── 4_Risk_Panel.py
├── requirements.txt
└── README.md
```

---

## Key Design Decisions

**Lead Time Engineering**: The raw dataset's ship dates are placeholder values. Lead time was engineered from operational factors that actually drive it: ship mode base days + distance effect + realistic noise. This produces a meaningful 1–14 day range for the model to learn from.

**Modular Architecture**: Data download, preprocessing, feature engineering, model training, and optimization are fully separated, making each stage independently testable and replaceable.

**Scenario Simulation**: For every product, the optimizer simulates assignment to all 5 factories, predicts lead time for each, and ranks alternatives by operational impact — giving decision-makers a ranked action list rather than raw data.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data | pandas, numpy, geopy |
| ML | scikit-learn (LinearRegression, RandomForest, GradientBoosting) |
| Visualisation | Plotly |
| Dashboard | Streamlit |
| Deployment | Streamlit Community Cloud |
| Data Access | gdown (Google Drive) |
