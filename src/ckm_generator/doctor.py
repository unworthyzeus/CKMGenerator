"""Print CKM Generator runtime diagnostics."""
from __future__ import annotations

import argparse
import json

from .diagnostics import inspect_runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "directml", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    print(json.dumps(inspect_runtime(args.device).as_dict(), indent=2))


if __name__ == "__main__":
    main()
