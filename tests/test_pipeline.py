import pandas as pd

from model.data import clean, quality_report
from model.features import build_features, haversine_km
from model.schema import FEATURES
from model.split import time_split


def test_quality_report_counts(raw_sample):
    rep = quality_report(raw_sample)
    assert rep["rows"] == 5
    assert rep["fraud_count"] == 1
    assert rep["unique_cards"] == 2


def test_clean_drops_pii_and_bad_rows(raw_sample):
    bad = raw_sample.copy()
    bad.loc[0, "amt"] = -5
    out = clean(bad)
    assert len(out) == 4
    assert "first" not in out and "street" not in out


def test_features_only_use_past(raw_sample):
    feats = build_features(clean(raw_sample))
    assert set(FEATURES) <= set(feats.columns)
    card_a = feats[feats.cc_num == 111].reset_index(drop=True)
    # first txn has no history
    assert card_a.loc[0, "txn_count_1h"] == 0
    assert card_a.loc[0, "secs_since_last_txn"] == -1
    assert card_a.loc[0, "amt_to_card_mean"] == 1.0
    # second txn sees exactly one earlier txn in the last hour
    assert card_a.loc[1, "txn_count_1h"] == 1
    assert card_a.loc[1, "secs_since_last_txn"] == 1800
    # third txn: 2 txns in prior 24h, mean of past = 100 -> ratio 9
    assert card_a.loc[2, "txn_count_24h"] == 2
    assert card_a.loc[2, "amt_sum_24h"] == 200
    assert card_a.loc[2, "amt_to_card_mean"] == 9.0
    # card B's night transaction
    card_b = feats[feats.cc_num == 222].reset_index(drop=True)
    assert card_b.loc[1, "is_night"] == 1


def test_haversine_known_distance():
    # Norman, OK -> Oklahoma City, OK is roughly 30 km
    assert 25 < haversine_km(35.22, -97.44, 35.47, -97.52) < 35


def test_time_split_is_chronological(raw_sample):
    feats = build_features(clean(raw_sample))
    early, late, cutoff = time_split(feats, val_fraction=0.4)
    assert early.trans_date_trans_time.max() < late.trans_date_trans_time.min()
    assert len(early) + len(late) == len(feats)
    assert isinstance(cutoff, pd.Timestamp)
