"""
Phase 3: build X/y and a sklearn preprocessor (numeric + one-hot categoricals).

No leakage: `severity_class` is never included in X — only `FEATURE_COLUMNS` from utils.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import DEFAULT_PROCESSED_CSV, FEATURE_COLUMNS, TARGET_COLUMN

NUMERIC_FEATURES = ["hour", "day_of_week"]
CATEGORICAL_FEATURES = ["weather", "area_type", "traffic_level"]

# Sanity check: columns line up with MVP feature list
assert set(NUMERIC_FEATURES + CATEGORICAL_FEATURES) == set(FEATURE_COLUMNS)


def make_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split cleaned table into feature matrix X and target y."""
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """
    Encode numerics (impute + scale) and categoricals (impute + one-hot).

    OneHotEncoder uses handle_unknown='ignore' so prediction rows with unseen
    categories do not crash (all-zero columns for that category).
    """
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.25,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Train/test split. Stratify when possible (recommended for small, multiclass y).

    If stratification is impossible (too few rows per class), falls back without stratify.
    """
    strat = y if stratify else None
    try:
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=strat,
        )
    except ValueError as e:
        print(f"Warning: stratified split failed ({e}); retrying without stratify.")
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )


def load_clean_table(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path is not None else DEFAULT_PROCESSED_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python scripts/run_phase2_eda.py"
        )
    return pd.read_csv(path)
