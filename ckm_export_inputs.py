from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parent


if __name__ == "__main__":
    runpy.run_path(str(ROOT / "scripts" / "export_input_samples.py"), run_name="__main__")
