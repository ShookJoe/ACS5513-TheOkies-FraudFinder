"""Central place for every file path the project uses."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PREPARED_DIR = DATA_DIR / "prepared"

RAW_TRAIN = RAW_DIR / "fraudTrain.csv"
RAW_TEST = RAW_DIR / "fraudTest.csv"

TRAIN_FILE = PREPARED_DIR / "train.csv"
VAL_FILE = PREPARED_DIR / "val.csv"
TEST_FILE = PREPARED_DIR / "test.csv"

STORAGE_DIR = ROOT / "storage"
MODELS_DIR = STORAGE_DIR / "models"
GRAPHICS_DIR = STORAGE_DIR / "graphics"
REPORTS_DIR = STORAGE_DIR / "reports"

KAGGLE_DATASET = "kartik2112/fraud-detection"
