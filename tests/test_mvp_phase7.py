"""
Phase 7: smoke tests — multiple scenarios, valid outputs, X column order vs training.

Run from project root (with venv):
  pip install -r requirements.txt
  python main.py train   # if models/accident_risk.joblib is missing
  pytest tests/test_mvp_phase7.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from predict import inputs_to_dataframe, load_pipeline, predict_scenario  # noqa: E402
from utils import DEFAULT_MODEL_PATH, FEATURE_COLUMNS  # noqa: E402

pytestmark = pytest.mark.skipif(
    not DEFAULT_MODEL_PATH.exists(),
    reason="No trained model — run: python main.py train",
)


# Diverse scenarios: clear day urban, rain+night+highway, fog rural, snow morning highway
SCENARIOS = [
    {"hour": 10, "day_of_week": 2, "weather": "Clear", "area_type": "Urban", "traffic_level": "Low"},
    {"hour": 22, "day_of_week": 5, "weather": "Rain", "area_type": "Highway", "traffic_level": "High"},
    {"hour": 14, "day_of_week": 6, "weather": "Fog", "area_type": "Rural", "traffic_level": "Medium"},
    {"hour": 6, "day_of_week": 0, "weather": "Snow", "area_type": "Highway", "traffic_level": "Low"},
    {"hour": 18, "day_of_week": 4, "weather": "Clear", "area_type": "Urban", "traffic_level": "High"},
]


def test_inputs_dataframe_matches_training_columns() -> None:
    """Guards the common bug: prediction row must use same column names/order as training X."""
    df = inputs_to_dataframe(12, 3, "Rain", "Urban", "Medium")
    assert list(df.columns) == list(FEATURE_COLUMNS)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: f"h{s['hour']}_{s['weather']}_{s['area_type']}")
def test_predict_scenario_outputs(scenario: dict) -> None:
    pipe = load_pipeline()
    out = predict_scenario(pipe, **scenario)
    assert out["risk_level"] in {"Low", "Medium", "High"}
    assert out["severity_class"] in {0, 1, 2}
    assert isinstance(out["recommendation"], str)
    assert len(out["recommendation"].strip()) > 0


def test_pipeline_predict_accepts_feature_frame() -> None:
    """End-to-end shape check through the saved sklearn pipeline."""
    pipe = load_pipeline()
    X = inputs_to_dataframe(15, 1, "Clear", "Rural", "Medium")
    preds = pipe.predict(X)
    assert preds.shape == (1,)
    assert int(preds[0]) in {0, 1, 2}
