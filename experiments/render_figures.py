from __future__ import annotations

from pathlib import Path

from experiments import exp1, exp2, exp3, exp4
from src.config import CACHE_DIR


def render_all() -> dict[str, Path]:
    """Refresh all figures/tables using the deterministic cached workflow.

    The experiment modules write their numeric outputs to outputs/cache and
    outputs/tables. Re-running this renderer is safe for style/report updates:
    it does not touch the input EEG files or PDFs, and it preserves the fixed
    random seed used for the artificial signals.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for module in (exp1, exp2, exp3, exp4):
        outputs.update(module.run())
    return outputs


if __name__ == "__main__":
    render_all()
