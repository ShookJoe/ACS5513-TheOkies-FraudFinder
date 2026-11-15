import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def raw_sample():
    """Tiny hand-made frame with the same columns as fraudTrain.csv."""
    rows = [
        # card A: three transactions, two within one hour
        ("2019-01-01 10:00:00", 111, "grocery_pos", 50.0, 0),
        ("2019-01-01 10:30:00", 111, "shopping_net", 150.0, 0),
        ("2019-01-02 09:00:00", 111, "misc_net", 900.0, 1),
        # card B
        ("2019-01-01 12:00:00", 222, "gas_transport", 20.0, 0),
        ("2019-01-01 23:30:00", 222, "travel", 40.0, 0),
    ]
    df = pd.DataFrame(rows, columns=["trans_date_trans_time", "cc_num", "category", "amt", "is_fraud"])
    df["Unnamed: 0"] = range(len(df))
    df["merchant"] = "fraud_Shop"
    df["first"], df["last"], df["street"] = "Jane", "Doe", "1 Main St"
    df["city"], df["state"], df["zip"], df["job"] = "Norman", "OK", 73019, "Engineer"
    df["gender"] = ["F", "F", "F", "M", "M"]
    df["lat"], df["long"] = 35.22, -97.44
    df["merch_lat"], df["merch_long"] = 35.47, -97.52
    df["city_pop"] = 128000
    df["dob"] = "1990-06-15"
    df["trans_num"] = [f"t{i}" for i in range(len(df))]
    df["unix_time"] = 0
    return df
