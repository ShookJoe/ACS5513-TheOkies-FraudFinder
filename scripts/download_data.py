"""Download the Sparkov credit-card fraud dataset from Kaggle into data/raw/.

Setup (once per teammate):
  1. Kaggle -> Settings -> API -> "Create New Token" (downloads kaggle.json)
  2. Put kaggle.json in  %USERPROFILE%\\.kaggle\\  (Windows) or ~/.kaggle/ (Mac/Linux)
  3. python scripts/download_data.py

Never commit kaggle.json (it is in .gitignore).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.paths import KAGGLE_DATASET, RAW_DIR, RAW_TEST, RAW_TRAIN  # noqa: E402


def main():
    if RAW_TRAIN.exists() and RAW_TEST.exists():
        print(f"Data already present in {RAW_DIR}")
        return
    from kaggle.api.kaggle_api_extended import KaggleApi

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    print(f"Downloading {KAGGLE_DATASET} -> {RAW_DIR} (~200 MB zipped)...")
    api.dataset_download_files(KAGGLE_DATASET, path=str(RAW_DIR), unzip=True, quiet=False)
    print("Done:", sorted(p.name for p in RAW_DIR.iterdir()))


if __name__ == "__main__":
    main()
