from __future__ import annotations

from pathlib import Path

from src.config import ensure_output_dirs

from experiments import exp1, exp2, exp3, exp4


def run_all() -> dict[str, Path]:
    ensure_output_dirs()
    outputs: dict[str, Path] = {}
    for module in (exp1, exp2, exp3, exp4):
        outputs.update(module.run())
    return outputs


if __name__ == "__main__":
    run_all()
