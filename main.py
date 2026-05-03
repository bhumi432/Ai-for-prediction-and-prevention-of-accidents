"""
Phase 6: single entrypoint — train, predict (CLI), or launch Streamlit UI.

Examples:
  python main.py train
  python main.py predict --hour 14 --day-of-week 2 --weather Clear --area-type Urban --traffic-level Medium
  python main.py ui
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def cmd_ui() -> int:
    app = PROJECT_ROOT / "app_streamlit.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app)]
    print("Launching Streamlit (Ctrl+C to stop):", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Road accident risk MVP — train, predict, or open UI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("train", help="Train and save model (pass through flags, e.g. --model dt)")
    sub.add_parser(
        "predict",
        help="Print JSON prediction (pass through flags, e.g. --hour 20 --weather Rain ...)",
    )
    sub.add_parser("ui", help="Open Streamlit dashboard in the browser")

    args, remaining = parser.parse_known_args()

    if args.command == "train":
        from train import run_train

        return run_train(remaining)
    if args.command == "predict":
        from predict import run_predict

        return run_predict(remaining)
    if args.command == "ui":
        return cmd_ui()

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
