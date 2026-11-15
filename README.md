# Fraud Finder — The Okies

Credit-card fraud detection, built end-to-end for ACS 5513 following Microsoft's Team Data Science Process (TDSP).
Data: Kaggle [Credit Card Transactions Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection) (Sparkov, simulated).

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

## Get & prepare the data
1. Sign in to Kaggle once from the terminal (opens a browser): `.venv\Scripts\kaggle auth login`
   (Alternative: manually download the zip from the dataset page and extract both CSVs into `data/raw/`.)
2. Run:
```bash
python scripts/download_data.py   # -> data/raw/fraudTrain.csv, fraudTest.csv
python scripts/prepare_data.py    # -> data/prepared/{train,val,test}.csv + storage/reports/*.json
python scripts/eda.py             # -> storage/reports/eda_summary.md + storage/graphics/*.png
```

## Run the app / tests
```bash
python run.py          # http://127.0.0.1:5000
python -m pytest -q
```

## Repository layout
```
app/        Flask app (pages, JSON API, form -> features)
config/     paths
model/      schema, cleaning, feature engineering, splitting
scripts/    download, prepare, EDA (training added in Iteration 2)
storage/    generated reports, figures, models
tests/      pytest
```

## Splits
Chronological, never random. `train` = first 80% of fraudTrain, `val` = last 20% (used for tuning),
`test` = fraudTest (used once, at the end).
