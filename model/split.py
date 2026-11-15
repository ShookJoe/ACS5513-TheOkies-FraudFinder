"""Train / validation / test splitting.

Fraud data is a time series, so we split by TIME, never randomly:

    |------------- fraudTrain.csv -------------|---- fraudTest.csv ----|
    |-------- train (~80%) --------|-- val --- |-------- test ---------|
    Jan 2019                                 Jun 2020              Dec 2020

* train : fit models
* val   : compare models, tune hyper-parameters and the decision threshold
* test  : touched ONCE at the end (iteration 2/3) to report final numbers
"""
from __future__ import annotations

import pandas as pd

TIME_COL = "trans_date_trans_time"


def time_split(df: pd.DataFrame, val_fraction: float = 0.2):
    """Split a time-sorted frame into (earlier, later) by a time cutoff."""
    df = df.sort_values(TIME_COL, kind="mergesort")
    cutoff = df[TIME_COL].quantile(1 - val_fraction)
    early = df[df[TIME_COL] < cutoff].reset_index(drop=True)
    late = df[df[TIME_COL] >= cutoff].reset_index(drop=True)
    return early, late, cutoff
