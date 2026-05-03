"""
Phase 5: load trained pipeline, score one scenario, map to Low/Medium/High, add tips.

Usage:
  python predict.py --hour 22 --day-of-week 5 --weather Rain --area-type Highway --traffic-level High
  python main.py predict --hour 22 --day-of-week 5 --weather Rain --area-type Highway --traffic-level High
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from utils import DEFAULT_MODEL_PATH, FEATURE_COLUMNS

RISK_FROM_CLASS: dict[int, str] = {0: "Low", 1: "Medium", 2: "High"}


def load_pipeline(model_path: str | Path | None = None):
    path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Missing model at {path}. Train first: python train.py"
        )
    return joblib.load(path)


def inputs_to_dataframe(
    hour: int,
    day_of_week: int,
    weather: str,
    area_type: str,
    traffic_level: str,
) -> pd.DataFrame:
    """Single-row X with the same column order as training (`FEATURE_COLUMNS`)."""
    row: dict[str, Any] = {
        "hour": int(hour),
        "day_of_week": int(day_of_week),
        "weather": str(weather).strip(),
        "area_type": str(area_type).strip(),
        "traffic_level": str(traffic_level).strip(),
    }
    return pd.DataFrame([row])[FEATURE_COLUMNS]


def class_to_risk_level(pred_class: int) -> tuple[str, int]:
    """Map model integer label to display string; clamp unknown ints."""
    c = int(pred_class)
    if c not in RISK_FROM_CLASS:
        c = max(0, min(2, c))
    return RISK_FROM_CLASS[c], c


def build_recommendation(
    risk_level: str,
    *,
    weather: str,
    hour: int,
    area_type: str | None = None,
    traffic_level: str | None = None,
) -> str:
    """Rule-based safety text from risk + context (MVP)."""
    w = str(weather).lower()
    tips: list[str] = []

    if risk_level == "High":
        tips.append("Risk is high: delay the trip if you can, or choose a calmer route.")
    elif risk_level == "Medium":
        tips.append("Elevated risk: add extra margin and avoid rushing.")

    if any(x in w for x in ("rain", "snow", "fog", "storm", "ice")):
        tips.append("Wet or low-visibility conditions: slow down and increase following distance.")
    if hour >= 20 or hour <= 5:
        tips.append("Night driving: use proper lighting and scan farther ahead.")
    if area_type and str(area_type).lower() == "highway" and traffic_level:
        if str(traffic_level).lower() == "high":
            tips.append("Heavy traffic on highway: avoid sudden lane changes; watch merge zones.")

    if not tips:
        tips.append("Maintain a safe speed, avoid distractions, and keep space around your vehicle.")

    return " ".join(tips)


def predict_scenario(
    pipeline,
    hour: int,
    day_of_week: int,
    weather: str,
    area_type: str,
    traffic_level: str,
) -> dict[str, Any]:
    """Run model + recommendations; return a JSON-serializable dict."""
    X = inputs_to_dataframe(hour, day_of_week, weather, area_type, traffic_level)
    pred_class = int(pipeline.predict(X)[0])
    risk_level, cls = class_to_risk_level(pred_class)
    rec = build_recommendation(
        risk_level,
        weather=weather,
        hour=hour,
        area_type=area_type,
        traffic_level=traffic_level,
    )
    return {
        "risk_level": risk_level,
        "severity_class": cls,
        "recommendation": rec,
    }


def build_predict_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Predict accident risk (MVP)")
    p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    p.add_argument("--hour", type=int, required=True)
    p.add_argument("--day-of-week", type=int, required=True, help="0=Sun .. 6=Sat")
    p.add_argument("--weather", type=str, required=True)
    p.add_argument("--area-type", type=str, required=True, help="e.g. Urban, Highway, Rural")
    p.add_argument("--traffic-level", type=str, required=True, help="Low, Medium, High")
    return p


def run_predict(argv: list[str] | None = None) -> int:
    args = build_predict_parser().parse_args(argv)

    try:
        pipe = load_pipeline(args.model_path)
    except FileNotFoundError as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 1

    out = predict_scenario(
        pipe,
        hour=args.hour,
        day_of_week=args.day_of_week,
        weather=args.weather,
        area_type=args.area_type,
        traffic_level=args.traffic_level,
    )
    print(json.dumps(out, indent=2))
    return 0


def main() -> int:
    return run_predict()


if __name__ == "__main__":
    raise SystemExit(main())
