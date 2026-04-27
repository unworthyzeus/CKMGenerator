"""Project paths for the standalone CKM Generator."""
from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "generator_config.yaml"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
