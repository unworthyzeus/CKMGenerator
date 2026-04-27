#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from submit_try80_1gpu import main  # noqa: E402


if __name__ == "__main__":
    sys.argv.extend(["--slurm-file", "cluster/run_eightieth_try80_2gpu.slurm", "--job-name", "t80-2gpu"])
    main()
