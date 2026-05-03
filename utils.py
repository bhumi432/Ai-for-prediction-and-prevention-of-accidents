"""
Phase 2: load accident CSV, basic EDA, light cleaning.

Target: `severity_class` — 0 = Low risk, 1 = Medium, 2 = High (for this MVP / sample data).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RAW_CSV = PROJECT_ROOT / "data" / "raw" / "sample_accidents.csv"
DEFAULT_PROCESSED_CSV = PROJECT_ROOT / "data" / "processed" / "clean_accidents.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "accident_risk.joblib"

# Model target (label)
TARGET_COLUMN = "severity_class"

# Input features used after Phase 3+ (documented here in Phase 2 for clarity)
FEATURE_COLUMNS = ["hour", "day_of_week", "weather", "area_type", "traffic_level"]

VALID_SEVERITY = {0, 1, 2}


def load_accidents_csv(path: str | Path | None = None) -> pd.DataFrame:
    """Load a road-accidents-style CSV into a DataFrame."""
    path = Path(path) if path is not None else DEFAULT_RAW_CSV
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def print_basic_eda(df: pd.DataFrame, title: str = "Dataset") -> None:
    """Print shape, dtypes, missing counts, and target distribution."""
    print(f"\n=== {title} ===")
    print("shape:", df.shape)
    print("\nhead():\n", df.head())
    print("\ndtypes:\n", df.dtypes)
    print("\nmissing per column:\n", df.isnull().sum())
    if TARGET_COLUMN in df.columns:
        print(f"\n{TARGET_COLUMN} value counts:\n", df[TARGET_COLUMN].value_counts(dropna=False).sort_index())


def clean_accidents_df(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COLUMN,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Light cleaning for MVP tabular data:

    - Strip whitespace on string columns.
    - Coerce `hour`, `day_of_week`, and target to numeric; drop rows with invalid target.
    - Clip hour to [0, 23]; clip day_of_week to [0, 6] (0 = Sunday convention in sample).
    - Drop rows with missing values in target or feature columns (simplest baseline; no imputation yet).

    Documented choice: we **drop** incomplete rows instead of imputing in Phase 2 so Phase 3+ sees a strict rectangle.
    """
    feature_cols = feature_cols or FEATURE_COLUMNS
    out = df.copy()

    for col in out.select_dtypes(include="object").columns:
        out[col] = out[col].astype(str).str.strip()

    out["hour"] = pd.to_numeric(out["hour"], errors="coerce")
    out["day_of_week"] = pd.to_numeric(out["day_of_week"], errors="coerce")
    out[target_col] = pd.to_numeric(out[target_col], errors="coerce")

    out = out.dropna(subset=[target_col] + [c for c in feature_cols if c in out.columns])

    invalid_target = ~out[target_col].isin(list(VALID_SEVERITY))
    if invalid_target.any():
        n = int(invalid_target.sum())
        print(f"Note: dropping {n} row(s) with {target_col} not in {sorted(VALID_SEVERITY)}")
        out = out.loc[~invalid_target]

    out["hour"] = out["hour"].clip(0, 23)
    out["day_of_week"] = out["day_of_week"].clip(0, 6)

    out[target_col] = out[target_col].astype(int)
    out = out.dropna(subset=feature_cols)

    return out.reset_index(drop=True)


def save_processed(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved: {path}  (rows={len(df)})")
