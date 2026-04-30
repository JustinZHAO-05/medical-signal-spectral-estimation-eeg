from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

from .config import FIGURE_DIR


def setup_matplotlib() -> None:
    font_paths = [
        Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    ]
    for font_path in font_paths:
        if font_path.exists():
            try:
                font_manager.fontManager.addfont(str(font_path))
            except Exception:
                pass

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Noto Sans SC",
                "Microsoft YaHei",
                "SimHei",
                "Arial",
            ],
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.24,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.formatter.useoffset": False,
            "axes.formatter.use_mathtext": False,
            "axes.formatter.limits": (-20, 20),
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "lines.linewidth": 1.8,
        }
    )


def save_figure(fig: plt.Figure, filename: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURE_DIR / filename
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def finish_psd_axes(ax, xlim=(0, 40), ylabel="功率谱密度 (dB/Hz)") -> None:
    ax.set_xlim(*xlim)
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.24)


def add_band(ax, lo: float, hi: float, label: str, color: str = "#8ecae6") -> None:
    ax.axvspan(lo, hi, color=color, alpha=0.15, lw=0)
    y0, y1 = ax.get_ylim()
    ax.text((lo + hi) / 2, y1, label, ha="center", va="top", fontsize=9, color="#32566b")
