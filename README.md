# Nassau Candy Distributor — Factory Reallocation & Shipping Optimization System

A machine learning-powered decision intelligence system that recommends optimal factory assignments for Nassau Candy products to minimize shipping lead times and protect profit margins.

---

## Project Structure

```
nassau-candy-optimizer/
├── data/                        # Raw + processed data + model artifacts (auto-generated)
├── notebooks/
│   └── 01_eda.ipynb             # Annotated EDA notebook
├── src/
│   ├── download_data.py         # Downloads dataset from Google Drive
│   ├── preprocessing.py         # Cleans data, adds lead time + distances
│   ├── feature_engineering.py   # Encodes features for ML
│   ├── model_train.py           # Trains 3 models, saves best
│   └── optimizer.py             # Simulation + recommendation engine
├── streamlit_app/
│   ├── app.py                   # Landing page
│   └── pages/
│       ├── 1_Factory_Optimizer.py
│       ├── 2_WhatIf_Analysis.py
│       ├── 3_Recommendations.py
│       └── 4_Risk_Panel.py
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Create and activate virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline (in order)
```bash
python src/download_data.py
python src/preprocessing.py
python src/model_train.py
```

### 4. Launch the dashboard
```bash
streamlit run streamlit_app/app.py
```

---

## Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub (include the `data/` folder with generated artifacts)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo, set main file: `streamlit_app/app.py`
4. Deploy

---

## Models Used
- Linear Regression (baseline)
- Random Forest Regressor
- Gradient Boosting Regressor

Best model is auto-selected by lowest RMSE and saved to `data/best_model.pkl`.

---

## Dataset
Nassau Candy Distributor order data — products, factories, shipping modes, dates, sales, costs, and profit.
