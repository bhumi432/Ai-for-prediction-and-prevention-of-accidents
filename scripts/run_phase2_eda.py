"""
Phase 2 entrypoint: load CSV, print EDA, clean, optionally save processed file.

Usage (from project root, venv activated):
  python scripts/run_phase2_eda.py
  python scripts/run_phase2_eda.py --input data/raw/sample_accidents.csv --output data/processed/clean_accidents.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/run_phase2_eda.py` from project root
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils import (  # noqa: E402
    DEFAULT_PROCESSED_CSV,
    DEFAULT_RAW_CSV,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    clean_accidents_df,
    load_accidents_csv,
    print_basic_eda,
    save_processed,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Phase 2: load data, EDA, clean")
    p.add_argument("--input", type=Path, default=DEFAULT_RAW_CSV, help="Path to raw CSV")
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PROCESSED_CSV,
        help="Path for cleaned CSV (default: data/processed/clean_accidents.csv)",
    )
    p.add_argument("--no-save", action="store_true", help="Do not write processed CSV")
    args = p.parse_args()

    print("Target column:", TARGET_COLUMN)
    print(
        "Rationale: severity_class encodes accident / risk severity as Low(0), Medium(1), High(2) "
        "for supervised learning on the sample dataset."
    )
    print("Feature columns (for later phases):", FEATURE_COLUMNS)

    raw = load_accidents_csv(args.input)
    print_basic_eda(raw, title="Raw")

    cleaned = clean_accidents_df(raw)
    print_basic_eda(cleaned, title="After cleaning")

    if not args.no_save:
        out_path = args.output if args.output is not None else DEFAULT_PROCESSED_CSV
        save_processed(cleaned, out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
