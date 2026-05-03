"""Phase 1 exit check: run inside your activated venv."""
import sys

def main() -> int:
    try:
        import pandas  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError as e:
        print("FAIL:", e, file=sys.stderr)
        print("Install deps: pip install -r requirements.txt", file=sys.stderr)
        return 1
    print("ok: pandas and scikit-learn import successfully")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
