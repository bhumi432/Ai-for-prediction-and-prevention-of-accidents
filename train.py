"""
Phase 4: train preprocess + classifier pipeline, evaluate on holdout, save with joblib.

Usage (from project root, venv activated):
  python train.py
  python main.py train
  python train.py --model dt --output models/accident_tree.joblib
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from preprocess import build_preprocessor, load_clean_table, make_xy, split_train_test
from utils import DEFAULT_MODEL_PATH, TARGET_COLUMN


def build_classifier(kind: str):
    kind = kind.lower()
    if kind == "lr":
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )
    if kind == "dt":
        return DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=42,
        )
    raise ValueError("classifier must be 'lr' or 'dt'")


def build_train_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train accident risk model")
    p.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Cleaned CSV (default: data/processed/clean_accidents.csv)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Where to save the fitted Pipeline (.joblib)",
    )
    p.add_argument(
        "--model",
        choices=("lr", "dt"),
        default="lr",
        help="lr = Logistic Regression, dt = Decision Tree",
    )
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--random-state", type=int, default=42)
    return p


def run_train(argv: list[str] | None = None) -> int:
    args = build_train_parser().parse_args(argv)

    df = load_clean_table(args.input)
    X, y = make_xy(df)
    X_train, X_test, y_train, y_test = split_train_test(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=True,
    )

    clf = build_classifier(args.model)
    pipeline = Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("classifier", clf),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print(f"Classifier: {args.model}")
    print("Test accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("\nClassification report (test):\n")
    # Labels 0,1,2 for readable report
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1, 2],
            target_names=["Low (0)", "Medium (1)", "High (2)"],
            zero_division=0,
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.output)
    print(f"Saved pipeline: {args.output}")

    # Smoke check: reload and predict one row
    loaded = joblib.load(args.output)
    sample = X_test.iloc[:1]
    p_one = loaded.predict(sample)[0]
    print(f"Smoke test: predict one holdout row -> {TARGET_COLUMN}={p_one}")

    return 0


def main() -> int:
    return run_train()


if __name__ == "__main__":
    raise SystemExit(main())
