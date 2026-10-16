from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("LEMON_EEG_DIR", PROJECT_ROOT / "data" / "sub-032481")).expanduser()
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
CACHE_DIR = OUTPUT_DIR / "cache"
PDF_DIR = OUTPUT_DIR / "pdf"

FS = 250.0
RANDOM_SEED = 20260430


def ensure_output_dirs() -> None:
    for path in (OUTPUT_DIR, FIGURE_DIR, TABLE_DIR, CACHE_DIR, PDF_DIR):
        path.mkdir(parents=True, exist_ok=True)


def next_pow_two(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()
