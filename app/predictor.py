"""Turns one transaction typed into the web form into a model prediction.

Iteration 1: the model does not exist yet, so `predict` reports that.
Iteration 2: train_model.py saves storage/models/fraud_model.joblib
             (an sklearn Pipeline) and this file starts using it.
"""
from __future__ import annotations

from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from config.paths import MODELS_DIR
from model.features import haversine_km
from model.schema import CATEGORIES, FEATURES

MODEL_PATH = MODELS_DIR / "fraud_model.joblib"
_model = None


def load_model():
    global _model
    if _model is None and MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
    return _model


def form_to_features(form: dict) -> pd.DataFrame:
    """Build the single-row feature frame the model expects.

    Card-history fields (txn counts, time since last txn, card average) can't
    be computed from one transaction, so the form asks for them directly;
    in a real bank system they would come from a feature store.
    """
    ts = datetime.fromisoformat(form["trans_datetime"])
    dob = datetime.fromisoformat(form["dob"])
    amt = float(form["amt"])
    card_mean = float(form.get("card_avg_amt") or amt)
    category = form["category"]
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category: {category}")

    row = {
        "amt": amt,
        "log_amt": np.log1p(amt),
        "hour": ts.hour,
        "day_of_week": ts.weekday(),
        "is_night": int(ts.hour >= 22 or ts.hour < 4),
        "age": round((ts - dob).days / 365.25, 1),
        "distance_km": float(haversine_km(float(form["lat"]), float(form["long"]),
                                          float(form["merch_lat"]), float(form["merch_long"]))),
        "log_city_pop": np.log1p(float(form["city_pop"])),
        "secs_since_last_txn": float(form.get("secs_since_last_txn") or -1),
        "txn_count_1h": float(form.get("txn_count_1h") or 0),
        "txn_count_24h": float(form.get("txn_count_24h") or 0),
        "amt_sum_24h": float(form.get("amt_sum_24h") or 0),
        "amt_to_card_mean": amt / card_mean if card_mean > 0 else 1.0,
        "category": category,
        "gender": form["gender"].upper(),
    }
    return pd.DataFrame([row])[FEATURES]


def predict(form: dict) -> dict:
    features = form_to_features(form)
    model = load_model()
    if model is None:
        return {"ready": False, "message": "Model not trained yet (coming in Iteration 2).",
                "features": features.iloc[0].to_dict()}
    proba = float(model.predict_proba(features)[0, 1])
    threshold = float(getattr(model, "threshold_", 0.5))
    return {"ready": True, "probability": proba, "threshold": threshold,
            "prediction": int(proba >= threshold), "features": features.iloc[0].to_dict()}
