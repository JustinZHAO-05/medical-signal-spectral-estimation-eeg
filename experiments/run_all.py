from __future__ import annotations

import argparse
from pathlib import Path

from src.config import ensure_output_dirs

from experiments import exp1, exp2, exp3, exp4


def run_all(*, synthetic_only: bool = False) -> dict[str, Path]:
    ensure_output_dirs()
    outputs: dict[str, Path] = {}
    for module in (exp1, exp2, exp3, exp4):
        outputs.update(module.run(synthetic_only=synthetic_only))
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spectral estimation experiments")
    parser.add_argument("--synthetic-only", action="store_true", help="Run without external EEG data")
    args = parser.parse_args()
    run_all(synthetic_only=args.synthetic_only)
