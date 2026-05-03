"""
Phase 3: build X/y, fit preprocessor on train only, print shapes and feature names.

Usage (project root, venv on):
  python scripts/run_phase3_features.py
  python scripts/run_phase3_features.py --input data/processed/clean_accidents.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from preprocess import (  # noqa: E402
    build_preprocessor,
    load_clean_table,
    make_xy,
    split_train_test,
)
from utils import FEATURE_COLUMNS, TARGET_COLUMN  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Phase 3: features + train/test split")
    p.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Cleaned CSV (default: data/processed/clean_accidents.csv)",
    )
    args = p.parse_args()

    df = load_clean_table(args.input)
    X, y = make_xy(df)

    print("Input feature columns (raw, for prediction rows):", FEATURE_COLUMNS)
    print("Target column:", TARGET_COLUMN)
    print("X shape:", X.shape, "| y shape:", y.shape)
    print("y class counts:\n", y.value_counts().sort_index())

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    print("\n--- Train/test ---")
    print("X_train:", X_train.shape, "X_test:", X_test.shape)
    print("y_train counts:\n", y_train.value_counts().sort_index())
    print("y_test counts:\n", y_test.value_counts().sort_index())

    prep = build_preprocessor()
    prep.fit(X_train)
    Xt_train = prep.transform(X_train)
    Xt_test = prep.transform(X_test)

    print("\n--- After preprocessing ---")
    print("Transformed X_train shape:", Xt_train.shape)
    print("Transformed X_test shape:", Xt_test.shape)
    names = prep.get_feature_names_out()
    print("Number of engineered features:", len(names))
    print("First 15 feature names:", list(names[:15]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
