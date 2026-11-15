"""Feature engineering.

Every behavioural feature looks ONLY at a card's earlier transactions, so the
model never sees information from the future (no leakage). None of the
features use the fraud label.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points, in km (vectorised)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features computed from a single row (no history needed)."""
    out = df.copy()
    ts = out["trans_date_trans_time"]
    out["log_amt"] = np.log1p(out["amt"])
    out["hour"] = ts.dt.hour
    out["day_of_week"] = ts.dt.day_of_week
    out["is_night"] = ((out["hour"] >= 22) | (out["hour"] < 4)).astype(int)
    out["age"] = ((ts - out["dob"]).dt.days / 365.25).round(1)
    out["distance_km"] = haversine_km(out["lat"], out["long"], out["merch_lat"], out["merch_long"])
    out["log_city_pop"] = np.log1p(out["city_pop"])
    return out


def add_card_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-card 'velocity' features using only PAST transactions of that card."""
    out = df.sort_values(["cc_num", "trans_date_trans_time"], kind="mergesort").reset_index(drop=True)
    g = out.groupby("cc_num", sort=False)

    # Time since the card's previous transaction (first transaction -> -1).
    prev_ts = g["trans_date_trans_time"].shift()
    out["secs_since_last_txn"] = (
        (out["trans_date_trans_time"] - prev_ts).dt.total_seconds().fillna(-1)
    )

    # Rolling counts / sums over previous 1h and 24h (closed='left' excludes current row).
    rolled = out.set_index("trans_date_trans_time").groupby("cc_num", sort=False)["amt"]
    out["txn_count_1h"] = rolled.rolling("1h", closed="left").count().fillna(0).to_numpy()
    out["txn_count_24h"] = rolled.rolling("24h", closed="left").count().fillna(0).to_numpy()
    out["amt_sum_24h"] = rolled.rolling("24h", closed="left").sum().fillna(0).to_numpy()

    # Current amount relative to the card's average of all PAST amounts.
    past_sum = g["amt"].cumsum() - out["amt"]
    past_n = g.cumcount()
    past_mean = past_sum / past_n.replace(0, np.nan)
    out["amt_to_card_mean"] = (out["amt"] / past_mean).fillna(1.0)

    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run all feature steps. Input = output of model.data.clean()."""
    out = add_basic_features(df)
    out = add_card_history_features(out)
    return out.sort_values("trans_date_trans_time", kind="mergesort").reset_index(drop=True)
