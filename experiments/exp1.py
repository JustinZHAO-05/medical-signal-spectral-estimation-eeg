from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CACHE_DIR, FIGURE_DIR, FS, RANDOM_SEED, TABLE_DIR, ensure_output_dirs, next_pow_two
from src.data_io import load_eeg
from src.plotting import add_band, finish_psd_axes, save_figure, setup_matplotlib
from src.spectral_methods import (
    alpha_metrics,
    autocorrelation_psd,
    band_mean_db,
    band_peak,
    direct_psd,
    periodogram_psd,
    roughness_db,
    welch_psd,
)


def run() -> dict[str, Path]:
    ensure_output_dirs()
    setup_matplotlib()
    outputs: dict[str, Path] = {}
    outputs.update(_artificial_methods())
    outputs.update(_real_eeg_methods())
    return outputs


def _artificial_methods() -> dict[str, Path]:
    rng = np.random.default_rng(RANDOM_SEED + 1)
    fs = FS
    t = np.arange(0.0, 4.0, 1.0 / fs)
    x = np.sin(2 * np.pi * 12.0 * t) + 0.6 * np.sin(2 * np.pi * 24.0 * t) + 0.8 * rng.standard_normal(len(t))
    nfft = 4096

    fig, ax = plt.subplots(figsize=(9.4, 3.4))
    ax.plot(t, x, color="#264653", lw=1.0)
    ax.set_title("实验一(1)：人工复合信号时域波形")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("幅值")
    ax.set_xlim(0, 4)
    ax.grid(True, alpha=0.24)
    time_fig_path = save_figure(fig, "exp1_artificial_time_signal.png")

    psds = [
        direct_psd(x, fs, nfft=nfft, label="直接法(矩形窗FFT)"),
        periodogram_psd(x, fs, window="hamming", nfft=nfft, label="周期图法(Hamming窗)"),
        autocorrelation_psd(x, fs, nfft=nfft, label="自相关法"),
        welch_psd(x, fs, segment_s=1.0, overlap=0.5, window="hamming", nfft=nfft, label="Welch法(1s/50%)"),
    ]

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    for psd in psds:
        ax.plot(psd.freqs, psd.db, label=psd.label)
    ax.axvline(12, color="#d62828", ls="--", lw=1.2, label="12 Hz")
    ax.axvline(24, color="#f77f00", ls="--", lw=1.2, label="24 Hz")
    ax.set_title("实验一(1)：人工复合信号的四种非参数谱估计")
    finish_psd_axes(ax, xlim=(0, 45))
    ax.legend(ncol=2, fontsize=9)
    fig_path = save_figure(fig, "exp1_artificial_methods.png")

    rows = []
    for psd in psds:
        db = psd.db
        f12, p12 = band_peak(psd.freqs, db, (10.0, 14.0))
        f24, p24 = band_peak(psd.freqs, db, (22.0, 26.0))
        bg = band_mean_db(psd.freqs, db, (30.0, 45.0))
        bg_15_30 = band_mean_db(psd.freqs, db, (15.0, 30.0))
        rows.append(
            {
                "method": psd.label,
                "peak_near_12_hz": f12,
                "peak_12_db_hz": p12,
                "peak_near_24_hz": f24,
                "peak_24_db_hz": p24,
                "harmonic_24_prominence_vs_15_30_db": p24 - bg_15_30,
                "background_30_45_mean_db_hz": bg,
                "roughness_5_40_db": roughness_db(psd.freqs, db, (5.0, 40.0)),
            }
        )
    table_path = TABLE_DIR / "exp1_artificial_metrics.csv"
    pd.DataFrame(rows).to_csv(table_path, index=False, encoding="utf-8-sig")
    np.savez_compressed(CACHE_DIR / "exp1_artificial_psd.npz", x=x, t=t, **_psds_to_npz(psds))
    return {"exp1_artificial_time_figure": time_fig_path, "exp1_artificial_figure": fig_path, "exp1_artificial_table": table_path}


def _real_eeg_methods() -> dict[str, Path]:
    eeg = load_eeg()
    fs = eeg.fs
    x = eeg.segment("Oz", 58.84, 117.84)
    t = np.arange(len(x)) / fs + 58.84
    nfft = next_pow_two(2 * len(x))

    fig, ax = plt.subplots(figsize=(9.4, 3.4))
    ax.plot(t, x * 1e6, color="#264653", lw=0.65)
    ax.set_title("实验一(2)：真实Oz闭眼EEG 59 s分析片段时域波形")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("电位 (µV)")
    ax.set_xlim(t[0], t[-1])
    ax.grid(True, alpha=0.24)
    time_fig_path = save_figure(fig, "exp1_real_eeg_time_segment.png")

    psds = [
        direct_psd(x, fs, nfft=nfft, label="直接法(矩形窗FFT)"),
        periodogram_psd(x, fs, window="hamming", nfft=nfft, label="周期图法(Hamming窗)"),
        autocorrelation_psd(x, fs, nfft=nfft, label="自相关法"),
        welch_psd(x, fs, segment_s=1.0, overlap=0.5, window="hamming", nfft=nfft, label="Welch法(1s/50%)"),
    ]

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    for psd in psds:
        ax.plot(psd.freqs, psd.db, label=psd.label)
    add_band(ax, 8, 13, "α频段")
    ax.set_title("实验一(2)：Oz导联59 s闭眼EEG的经典谱估计比较")
    finish_psd_axes(ax, xlim=(0, 40))
    ax.legend(ncol=2, fontsize=9)
    fig_path = save_figure(fig, "exp1_real_eeg_methods.png")

    rows = []
    for psd in psds:
        row = {"method": psd.label}
        row.update(alpha_metrics(psd.freqs, psd.db))
        rows.append(row)
    table_path = TABLE_DIR / "exp1_real_alpha_metrics.csv"
    pd.DataFrame(rows).to_csv(table_path, index=False, encoding="utf-8-sig")

    summary_path = CACHE_DIR / "eeg_load_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "source": eeg.source,
                "shape": list(eeg.X.shape),
                "fs": eeg.fs,
                "n_channels": len(eeg.ch_names),
                "oz_index": eeg.ch_names.index("Oz"),
                "unit": eeg.unit,
                "scale_factor": eeg.scale_factor,
                "segment": "58.84-117.84 s",
                "segment_samples": len(x),
                "oz_segment_std_v": float(np.std(x)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    np.savez_compressed(CACHE_DIR / "exp1_real_psd.npz", x=x, **_psds_to_npz(psds))
    return {"exp1_real_time_figure": time_fig_path, "exp1_real_figure": fig_path, "exp1_real_table": table_path}


def _psds_to_npz(psds):
    data = {}
    for i, psd in enumerate(psds):
        data[f"freqs_{i}"] = psd.freqs
        data[f"pxx_{i}"] = psd.pxx
        data[f"label_{i}"] = psd.label
    return data


if __name__ == "__main__":
    run()
