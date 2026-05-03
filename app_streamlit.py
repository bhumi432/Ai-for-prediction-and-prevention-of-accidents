"""
Streamlit UI for accident risk MVP (Phase 6).

Run from project root:
  streamlit run app_streamlit.py
  # or: python main.py ui
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from predict import predict_scenario  # noqa: E402
from utils import DEFAULT_MODEL_PATH  # noqa: E402

WEATHER_OPTIONS = ["Clear", "Rain", "Fog", "Snow"]
AREA_OPTIONS = ["Urban", "Highway", "Rural"]
TRAFFIC_OPTIONS = ["Low", "Medium", "High"]
DAY_NAMES = [
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
]


@st.cache_resource
def _load_pipeline():
    from predict import load_pipeline

    return load_pipeline()


def main() -> None:
    st.set_page_config(page_title="Accident risk", page_icon="🚗", layout="centered")
    st.title("Road accident risk (MVP)")
    st.caption("Educational demo — not for safety-critical decisions. Train first: `python main.py train`")

    if not DEFAULT_MODEL_PATH.exists():
        st.error(
            f"No model at `{DEFAULT_MODEL_PATH}`. Run **`python main.py train`** then refresh this page."
        )
        return

    hour = st.slider("Hour of day (0–23)", 0, 23, 14)
    day_of_week = st.selectbox(
        "Day of week (0 = Sunday)",
        options=list(range(7)),
        index=2,
        format_func=lambda i: f"{DAY_NAMES[i]} ({i})",
    )
    weather = st.selectbox("Weather", WEATHER_OPTIONS)
    area_type = st.selectbox("Area type", AREA_OPTIONS)
    traffic_level = st.selectbox("Traffic level", TRAFFIC_OPTIONS)

    if st.button("Predict risk", type="primary"):
        try:
            pipe = _load_pipeline()
        except FileNotFoundError as e:
            st.error(str(e))
            return
        out = predict_scenario(
            pipe,
            hour=hour,
            day_of_week=day_of_week,
            weather=weather,
            area_type=area_type,
            traffic_level=traffic_level,
        )
        level = out["risk_level"]
        if level == "High":
            st.error(f"**Risk level: {level}**")
        elif level == "Medium":
            st.warning(f"**Risk level: {level}**")
        else:
            st.success(f"**Risk level: {level}**")
        st.metric("Model class (0=Low, 1=Med, 2=High)", out["severity_class"])
        st.subheader("Suggestions")
        st.write(out["recommendation"])


if __name__ == "__main__":
    main()
