"""End-to-end data preparation:  raw CSV  ->  cleaned + engineered CSV splits.

    python scripts/prepare_data.py

Outputs
  data/prepared/{train,val,test}.csv
  storage/reports/data_quality.json      (numbers for the Data Report)
  storage/reports/split_summary.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from config.paths import (PREPARED_DIR, RAW_TEST, RAW_TRAIN, REPORTS_DIR,  # noqa: E402
                          TEST_FILE, TRAIN_FILE, VAL_FILE)
from model.data import clean, load_raw, quality_report  # noqa: E402
from model.features import build_features  # noqa: E402
from model.schema import FEATURES, TARGET  # noqa: E402
from model.split import TIME_COL, time_split  # noqa: E402


def summarize(name, df):
    return {
        "rows": int(len(df)),
        "fraud": int(df[TARGET].sum()),
        "fraud_rate": round(float(df[TARGET].mean()), 5),
        "start": str(df[TIME_COL].min()),
        "end": str(df[TIME_COL].max()),
    }


def main():
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw CSVs...")
    raw_train, raw_test = load_raw(RAW_TRAIN), load_raw(RAW_TEST)

    quality = {
        "train": quality_report(raw_train, "fraudTrain.csv"),
        "test": quality_report(raw_test, "fraudTest.csv"),
    }
    (REPORTS_DIR / "data_quality.json").write_text(json.dumps(quality, indent=2))
    print(json.dumps(quality, indent=2))

    print("Cleaning...")
    train, test = clean(raw_train), clean(raw_test)
    train["_source"], test["_source"] = "train", "test"

    # Card-history features are built on the combined timeline so that the
    # first test-period transactions still "remember" the card's earlier
    # activity. They only look backwards and never use the label.
    print("Engineering features...")
    full = build_features(pd.concat([train, test], ignore_index=True))

    keep = [TIME_COL, "cc_num"] + FEATURES + [TARGET]
    dev = full[full["_source"] == "train"][keep]
    test_df = full[full["_source"] == "test"][keep].reset_index(drop=True)
    train_df, val_df, cutoff = time_split(dev, val_fraction=0.2)

    train_df.to_csv(TRAIN_FILE, index=False)
    val_df.to_csv(VAL_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    split = {
        "val_cutoff": str(cutoff),
        "train": summarize("train", train_df),
        "val": summarize("val", val_df),
        "test": summarize("test", test_df),
    }
    (REPORTS_DIR / "split_summary.json").write_text(json.dumps(split, indent=2))
    print(json.dumps(split, indent=2))
    print(f"Saved splits to {PREPARED_DIR}")


if __name__ == "__main__":
    main()
