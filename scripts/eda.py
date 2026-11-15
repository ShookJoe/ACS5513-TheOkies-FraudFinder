"""Exploratory data analysis: summary statistics + figures for the Data Report.

    python scripts/eda.py          (run after prepare_data.py)

Outputs
  storage/reports/eda_summary.md
  storage/graphics/*.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from config.paths import GRAPHICS_DIR, REPORTS_DIR, TRAIN_FILE  # noqa: E402
from model.schema import NUMERIC_FEATURES, TARGET  # noqa: E402


def save(fig, name):
    fig.tight_layout()
    fig.savefig(GRAPHICS_DIR / name, dpi=120)
    plt.close(fig)


def main():
    GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(TRAIN_FILE)  # EDA on the training split only
    label = df[TARGET].map({0: "legit", 1: "fraud"})
    lines = ["# EDA summary (training split)\n"]

    # 1. Class balance
    counts = df[TARGET].value_counts().rename({0: "legit", 1: "fraud"})
    lines += ["## Class balance\n", counts.to_frame("count").to_markdown(), ""]
    fig, ax = plt.subplots(figsize=(4, 3))
    counts.plot.bar(ax=ax, color=["#4c72b0", "#c44e52"], logy=True)
    ax.set_title("Class balance (log scale)")
    save(fig, "class_balance.png")

    # 2. Numeric summary statistics by class
    stats = df.groupby(label)[NUMERIC_FEATURES].describe().T
    lines += ["## Numeric features by class\n", stats.round(2).to_markdown(), ""]

    # 3. Amount distribution
    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.histplot(data=df, x="log_amt", hue=label, stat="density", common_norm=False, bins=60, ax=ax)
    ax.set_title("log(1 + amount) by class")
    save(fig, "amount_by_class.png")

    # 4. Fraud rate by hour
    by_hour = df.groupby("hour")[TARGET].mean()
    lines += ["## Fraud rate by hour\n", by_hour.round(4).to_frame("fraud_rate").to_markdown(), ""]
    fig, ax = plt.subplots(figsize=(6, 3))
    by_hour.plot.bar(ax=ax, color="#c44e52")
    ax.set_title("Fraud rate by hour of day")
    save(fig, "fraud_rate_by_hour.png")

    # 5. Fraud rate by category
    by_cat = df.groupby("category")[TARGET].agg(["count", "mean"]).sort_values("mean", ascending=False)
    lines += ["## Fraud rate by category\n", by_cat.round(4).to_markdown(), ""]
    fig, ax = plt.subplots(figsize=(6, 4))
    by_cat["mean"].plot.barh(ax=ax, color="#c44e52")
    ax.set_title("Fraud rate by merchant category")
    save(fig, "fraud_rate_by_category.png")

    # 6. Correlation with target
    corr = df[NUMERIC_FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
    lines += ["## Correlation with is_fraud\n", corr.round(3).to_frame("corr").to_markdown(), ""]
    fig, ax = plt.subplots(figsize=(5, 4))
    corr.plot.barh(ax=ax)
    ax.set_title("Pearson correlation with is_fraud")
    save(fig, "feature_correlation.png")

    (REPORTS_DIR / "eda_summary.md").write_text("\n".join(lines))
    print(f"Wrote {REPORTS_DIR / 'eda_summary.md'} and figures in {GRAPHICS_DIR}")


if __name__ == "__main__":
    main()
