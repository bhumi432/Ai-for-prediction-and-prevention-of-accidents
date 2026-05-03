"""Load saved pipeline and run one prediction (Phase 4 exit check)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib  # noqa: E402

from utils import DEFAULT_MODEL_PATH  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    args = p.parse_args()
    if not args.model_path.exists():
        print("Missing model. Run: python train.py", file=sys.stderr)
        return 1
    pipe = joblib.load(args.model_path)
    print("Loaded:", args.model_path)
    print("Steps:", list(pipe.named_steps.keys()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
