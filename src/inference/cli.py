"""Command-line entry point for fixed-artifact NPP-Guard v1 inference."""

from __future__ import annotations

import argparse
import json

from .v1 import diagnose_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="Run NPP-Guard v1 fixed-artifact inference on one trajectory CSV.")
    parser.add_argument("--input", required=True, help="CSV path containing TIME and the strict 38 process variables")
    parser.add_argument("--artifacts", default=None, help="Optional artifacts/v1 directory")
    args = parser.parse_args()
    response = diagnose_csv(args.input, artifact_root=args.artifacts) if args.artifacts else diagnose_csv(args.input)
    print(json.dumps(response, ensure_ascii=False, indent=2, default=lambda value: value.item() if hasattr(value, "item") else str(value)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
