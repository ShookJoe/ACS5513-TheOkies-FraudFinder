"""Column definitions shared by data prep, training, and the Flask app.

Keeping these in one place means the app and the training code can never
disagree about which features the model expects.
"""

TARGET = "is_fraud"

# Raw columns we keep after cleaning (everything else is dropped).
RAW_KEEP = [
    "trans_date_trans_time", "cc_num", "category", "amt", "gender",
    "lat", "long", "city_pop", "dob", "merch_lat", "merch_long", TARGET,
]

# Personally identifying / free-text columns that are dropped on purpose.
# They either identify a person (privacy) or have too many unique values to
# generalize (merchant, job, street...). See docs/Project_Report_1.md.
DROPPED_COLUMNS = [
    "Unnamed: 0", "first", "last", "street", "city", "state", "zip",
    "job", "merchant", "trans_num", "unix_time",
]

CATEGORIES = [
    "entertainment", "food_dining", "gas_transport", "grocery_net",
    "grocery_pos", "health_fitness", "home", "kids_pets", "misc_net",
    "misc_pos", "personal_care", "shopping_net", "shopping_pos", "travel",
]

NUMERIC_FEATURES = [
    "amt",
    "log_amt",
    "hour",
    "day_of_week",
    "is_night",
    "age",
    "distance_km",
    "log_city_pop",
    "secs_since_last_txn",
    "txn_count_1h",
    "txn_count_24h",
    "amt_sum_24h",
    "amt_to_card_mean",
]

CATEGORICAL_FEATURES = ["category", "gender"]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
