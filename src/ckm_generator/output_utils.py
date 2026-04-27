"""Output directory helpers for CKM Generator."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

from .paths import DEFAULT_OUTPUT_DIR


def safe_output_name(value: str | None, *, fallback: str = "run") -> str:
    raw = (value or "").strip() or fallback
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._-")
    return safe or fallback


def make_timestamped_output_dir(*, base_dir: Path = DEFAULT_OUTPUT_DIR, label: str = "run") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(base_dir) / "runs" / f"{stamp}_{safe_output_name(label)}"


def resolve_output_dir(out_dir: Path | None, *, label: str = "run") -> Path:
    if out_dir is None:
        return make_timestamped_output_dir(label=label)
    return Path(out_dir).expanduser().resolve()


def list_existing_output_dirs(*, base_dir: Path = DEFAULT_OUTPUT_DIR) -> list[Path]:
    base = Path(base_dir)
    candidates: list[Path] = []
    for parent in (base, base / "runs", base / "validation"):
        if not parent.exists():
            continue
        for child in sorted(parent.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not child.is_dir():
                continue
            if child.name in {"_archived_tests", "logs"}:
                continue
            candidates.append(child)
    return candidates
