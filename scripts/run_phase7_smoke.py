"""
Phase 7: quick manual smoke (no pytest). Prints each scenario and PASS/FAIL.

Usage:
  python main.py train   # if needed
  python scripts/run_phase7_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from predict import inputs_to_dataframe, load_pipeline, predict_scenario  # noqa: E402
from utils import DEFAULT_MODEL_PATH, FEATURE_COLUMNS  # noqa: E402

SCENARIOS = [
    ("Clear daytime urban", dict(hour=10, day_of_week=2, weather="Clear", area_type="Urban", traffic_level="Low")),
    ("Rain + night + highway", dict(hour=22, day_of_week=5, weather="Rain", area_type="Highway", traffic_level="High")),
    ("Fog rural", dict(hour=14, day_of_week=6, weather="Fog", area_type="Rural", traffic_level="Medium")),
    ("Snow early highway", dict(hour=6, day_of_week=0, weather="Snow", area_type="Highway", traffic_level="Low")),
    ("Rush urban", dict(hour=18, day_of_week=4, weather="Clear", area_type="Urban", traffic_level="High")),
]


def main() -> int:
    print("=== Phase 7 smoke ===\n")
    if not DEFAULT_MODEL_PATH.exists():
        print("FAIL: no model at", DEFAULT_MODEL_PATH)
        print("Fix: python main.py train")
        return 1

    df = inputs_to_dataframe(1, 1, "Clear", "Urban", "Low")
    if list(df.columns) != list(FEATURE_COLUMNS):
        print("FAIL: column mismatch", list(df.columns), "vs", FEATURE_COLUMNS)
        return 1
    print("Column order OK:", FEATURE_COLUMNS)

    pipe = load_pipeline()
    failed = 0
    for label, kwargs in SCENARIOS:
        try:
            out = predict_scenario(pipe, **kwargs)
        except Exception as e:
            print(f"FAIL [{label}]: {e}")
            failed += 1
            continue
        ok = (
            out["risk_level"] in ("Low", "Medium", "High")
            and out["severity_class"] in (0, 1, 2)
            and len(str(out.get("recommendation", "")).strip()) > 0
        )
        status = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"{status} [{label}] -> {out['risk_level']} ({out['recommendation'][:60]}...)")

    print()
    if failed:
        print(f"Done: {failed} scenario(s) failed.")
        return 1
    print("Done: all scenarios passed.")
    print("\n--- Pre-demo checklist ---")
    print("[ ] python scripts/check_env.py")
    print("[ ] python main.py train   (if you changed data/features)")
    print("[ ] python scripts/run_phase7_smoke.py")
    print("[ ] pytest tests/test_mvp_phase7.py -v")
    print("[ ] python main.py predict --hour ... (spot-check)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
