"""Loading, data-quality checks, and cleaning of the raw Kaggle CSVs."""
from __future__ import annotations

import pandas as pd

from model.schema import RAW_KEEP, TARGET


def load_raw(path) -> pd.DataFrame:
    """Read one of the raw CSVs (fraudTrain.csv / fraudTest.csv)."""
    return pd.read_csv(path)


def quality_report(df: pd.DataFrame, name: str = "data") -> dict:
    """Summarise data-quality issues so they can go into the Data Report."""
    ts = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
    report = {
        "name": name,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_values": {c: int(n) for c, n in df.isna().sum().items() if n},
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_trans_num": int(df["trans_num"].duplicated().sum())
        if "trans_num" in df else None,
        "unparseable_timestamps": int(ts.isna().sum()),
        "date_min": str(ts.min()),
        "date_max": str(ts.max()),
        "non_positive_amounts": int((df["amt"] <= 0).sum()),
        "unique_cards": int(df["cc_num"].nunique()),
        "unique_merchants": int(df["merchant"].nunique()) if "merchant" in df else None,
        "fraud_count": int(df[TARGET].sum()),
        "fraud_rate": float(df[TARGET].mean()),
    }
    return report


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the useful columns, fix types, and drop bad rows."""
    out = df[RAW_KEEP].copy()
    out["trans_date_trans_time"] = pd.to_datetime(out["trans_date_trans_time"], errors="coerce")
    out["dob"] = pd.to_datetime(out["dob"], errors="coerce")
    out["category"] = out["category"].astype(str).str.strip().str.lower()
    out["gender"] = out["gender"].astype(str).str.strip().str.upper()

    out = out.dropna(subset=["trans_date_trans_time", "amt", "cc_num", TARGET])
    out = out[out["amt"] > 0]
    out = out.drop_duplicates()
    out[TARGET] = out[TARGET].astype(int)
    return out.reset_index(drop=True)
