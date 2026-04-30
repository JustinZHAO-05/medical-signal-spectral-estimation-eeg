from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CACHE_DIR, FS, TABLE_DIR, ensure_output_dirs, next_pow_two
from src.data_io import load_eeg
from src.plotting import add_band, finish_psd_axes, save_figure, setup_matplotlib
from src.spectral_methods import (
    alpha_metrics,
    band_mean_db,
    band_peak,
    count_local_maxima,
    minus_3db_width,
    periodogram_psd,
    rayleigh_interval,
    relative_valley_depth,
)


def run() -> dict[str, Path]:
    ensure_output_dirs()
    setup_matplotlib()
    outputs: dict[str, Path] = {}
    outputs.update(_length_resolution())
    outputs.update(_window_leakage())
    outputs.update(_nfft_illusion())
    outputs.update(_real_eeg_parameter_effects())
    return outputs


def _length_resolution() -> dict[str, Path]:
    fs = FS
    durations = [1.0, 2.0, 4.0, 8.0]
    nfft = 4096
    psds = []
    rows = []
    for duration in durations:
        t = np.arange(0.0, duration, 1.0 / fs)
        x = np.sin(2 * np.pi * 10.0 * t) + 0.8 * np.sin(2 * np.pi * 11.0 * t)
        psd = periodogram_psd(x, fs, window="boxcar", nfft=nfft, label=f"{duration:g} s")
        psds.append(psd)
        db = psd.db
        valley = relative_valley_depth(psd.freqs, db, (9.5, 10.5), (10.5, 11.5))
        apparent_peaks = count_local_maxima(psd.freqs, db, (9.0, 12.0), prominence_db=1.0, min_distance_hz=0.4)
        status = "临界可分" if duration <= 1.0 else "清晰可分"
        rows.append(
            {
                "duration_s": duration,
                "theoretical_delta_f_hz": rayleigh_interval(duration),
                "target_10_peak_hz": valley["left_peak_hz"],
                "target_11_peak_hz": valley["right_peak_hz"],
                "target_10_peak_db_hz": valley["left_peak_db_hz"],
                "target_11_peak_db_hz": valley["right_peak_db_hz"],
                "valley_between_peaks_db_hz": valley["valley_between_peaks_db_hz"],
                "weaker_peak_minus_valley_db": valley["weaker_peak_minus_valley_db"],
                "main_peak_minus_3db_width_hz": minus_3db_width(psd.freqs, db, (9.5, 10.5), (8.0, 12.5)),
                "apparent_local_peak_count_9_12": apparent_peaks,
                "resolution_status": status,
            }
        )

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for psd in psds:
        ax.plot(psd.freqs, psd.db, label=psd.label)
    ax.axvline(10, color="#d62828", ls="--", lw=1.1)
    ax.axvline(11, color="#f77f00", ls="--", lw=1.1)
    ax.set_title("实验二(1)：观测长度对10 Hz与11 Hz分辨能力的影响")
    finish_psd_axes(ax, xlim=(7, 14))
    ax.legend(title="数据长度")
    fig_path = save_figure(fig, "exp2_length_resolution.png")
    table_path = TABLE_DIR / "exp2_length_resolution_metrics.csv"
    pd.DataFrame(rows).to_csv(table_path, index=False, encoding="utf-8-sig")
    return {"exp2_length_figure": fig_path, "exp2_length_table": table_path}


def _window_leakage() -> dict[str, Path]:
    fs = FS
    duration = 1.8
    t = np.arange(0.0, duration, 1.0 / fs)
    x = np.sin(2 * np.pi * 10.3 * t) + 0.15 * np.sin(2 * np.pi * 14.1 * t)
    windows = [
        ("boxcar", "矩形窗"),
        ("hann", "Hann窗"),
        ("hamming", "Hamming窗"),
        ("blackman", "Blackman窗"),
    ]
    rows = []
    psds = []
    for window, label in windows:
        psd = periodogram_psd(x, fs, window=window, nfft=4096, label=label)
        psds.append(psd)
        db = psd.db
        f10, p10 = band_peak(psd.freqs, db, (9.0, 11.0))
        f14, p14 = band_peak(psd.freqs, db, (13.0, 15.0))
        bg = 0.5 * (band_mean_db(psd.freqs, db, (12.0, 13.0)) + band_mean_db(psd.freqs, db, (15.0, 16.0)))
        rows.append(
            {
                "window": label,
                "main_peak_hz": f10,
                "main_peak_db_hz": p10,
                "main_minus_3db_width_hz": minus_3db_width(psd.freqs, db, (9.0, 11.0), (7.0, 13.0)),
                "weak_peak_hz": f14,
                "weak_peak_db_hz": p14,
                "neighbor_background_db_hz": bg,
                "weak_peak_prominence_db": p14 - bg,
            }
        )

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for psd in psds:
        ax.plot(psd.freqs, psd.db, label=psd.label)
    ax.axvline(10.3, color="#d62828", ls="--", lw=1.1, label="10.3 Hz")
    ax.axvline(14.1, color="#2a9d8f", ls="--", lw=1.1, label="14.1 Hz")
    ax.set_title("实验二(2)：窗函数对谱泄漏和弱峰可见性的影响")
    finish_psd_axes(ax, xlim=(6, 18))
    ax.legend(ncol=2, fontsize=9)
    fig_path = save_figure(fig, "exp2_window_leakage.png")
    table_path = TABLE_DIR / "exp2_window_leakage_metrics.csv"
    pd.DataFrame(rows).to_csv(table_path, index=False, encoding="utf-8-sig")
    return {"exp2_window_figure": fig_path, "exp2_window_table": table_path}


def _nfft_illusion() -> dict[str, Path]:
    fs = FS
    t = np.arange(0.0, 1.0, 1.0 / fs)
    x = np.sin(2 * np.pi * 10.0 * t) + 0.8 * np.sin(2 * np.pi * 10.4 * t)
    nffts = [256, 2048, 8192]
    rows = []
    psds = []
    for nfft in nffts:
        psd = periodogram_psd(x, fs, window="boxcar", nfft=nfft, label=f"nfft={nfft}")
        psds.append(psd)
        apparent_peaks = count_local_maxima(psd.freqs, psd.db, (9.0, 12.0), prominence_db=1.0, min_distance_hz=0.25)
        rows.append(
            {
                "nfft": nfft,
                "frequency_grid_interval_hz": fs / nfft,
                "rayleigh_interval_hz": 1.0,
                "tone_spacing_hz": 0.4,
                "apparent_local_peak_count_9_12": apparent_peaks,
                "physically_resolvable_two_peaks": "否",
                "reason": "1 s观测长度的Rayleigh间隔约1 Hz，大于0.4 Hz频差；补零只加密频率采样点。",
            }
        )

    fig, axes = plt.subplots(len(psds), 1, figsize=(8.8, 7.6), sharex=True)
    for ax, psd in zip(axes, psds):
        ax.plot(psd.freqs, psd.db, color="#264653")
        ax.axvline(10.0, color="#d62828", ls="--", lw=1.0)
        ax.axvline(10.4, color="#f77f00", ls="--", lw=1.0)
        ax.set_title(f"{psd.label}, Δf={fs / int(psd.label.split('=')[1]):.4f} Hz", fontsize=10)
        finish_psd_axes(ax, xlim=(8.5, 12.2))
    axes[-1].set_xlabel("频率 (Hz)")
    fig.suptitle("实验二(3)：增大NFFT只加密频率轴，不能突破1 s观测长度的分辨率", y=0.995)
    fig_path = save_figure(fig, "exp2_nfft_subplots.png")
    table_path = TABLE_DIR / "exp2_nfft_metrics.csv"
    pd.DataFrame(rows).to_csv(table_path, index=False, encoding="utf-8-sig")
    return {"exp2_nfft_figure": fig_path, "exp2_nfft_table": table_path}


def _real_eeg_parameter_effects() -> dict[str, Path]:
    eeg = load_eeg()
    fs = eeg.fs
    base = eeg.segment("Oz", 58.84, 117.84)
    outputs: dict[str, Path] = {}

    # 1) Data length.
    rows = []
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for duration in [10.0, 30.0, 59.0]:
        n = int(round(duration * fs))
        x = base[:n]
        nfft = next_pow_two(len(x))
        psd = periodogram_psd(x, fs, window="hamming", nfft=nfft, label=f"{duration:g} s")
        ax.plot(psd.freqs, psd.db, label=f"{duration:g} s, nfft={nfft}")
        row = {"duration_s": duration, "nfft": nfft, "frequency_grid_interval_hz": fs / nfft}
        row["rayleigh_interval_hz"] = rayleigh_interval(duration)
        row.update(alpha_metrics(psd.freqs, psd.db))
        rows.append(row)
    add_band(ax, 8, 13, "α频段")
    ax.set_title("实验二(4-1)：真实Oz闭眼EEG中数据长度对α峰形态的影响")
    finish_psd_axes(ax, xlim=(2, 30))
    ax.legend(fontsize=9)
    outputs["exp2_real_length_figure"] = save_figure(fig, "exp2_real_length_alpha.png")
    table_path = TABLE_DIR / "exp2_real_length_alpha_metrics.csv"
    pd.DataFrame(rows).to_csv(table_path, index=False, encoding="utf-8-sig")
    outputs["exp2_real_length_table"] = table_path

    # 2) Window functions on the same 30 s segment.
    x30 = base[: int(round(30.0 * fs))]
    rows = []
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for window, label in [("boxcar", "矩形窗"), ("hamming", "Hamming窗"), ("blackman", "Blackman窗")]:
        psd = periodogram_psd(x30, fs, window=window, nfft=8192, label=label)
        ax.plot(psd.freqs, psd.db, label=label)
        row = {"window": label, "nfft": 8192}
        row.update(alpha_metrics(psd.freqs, psd.db))
        rows.append(row)
    add_band(ax, 8, 13, "α频段")
    ax.set_title("实验二(4-2)：真实Oz 30 s EEG中窗函数对α峰泄漏与宽度的影响")
    finish_psd_axes(ax, xlim=(2, 30))
    ax.legend(fontsize=9)
    outputs["exp2_real_window_figure"] = save_figure(fig, "exp2_real_windows_alpha.png")
    table_path = TABLE_DIR / "exp2_real_window_alpha_metrics.csv"
    pd.DataFrame(rows).to_csv(table_path, index=False, encoding="utf-8-sig")
    outputs["exp2_real_window_table"] = table_path

    # 3) NFFT on the same 30 s segment.
    rows = []
    nfft_psds = []
    for nfft in [7500, 15000, 30000]:
        psd = periodogram_psd(x30, fs, window="hamming", nfft=nfft, label=f"nfft={nfft}")
        nfft_psds.append((nfft, psd))
        row = {"nfft": nfft, "frequency_grid_interval_hz": fs / nfft}
        row.update(alpha_metrics(psd.freqs, psd.db))
        rows.append(row)
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 7.0), sharex=True, sharey=True)
    colors = {7500: "#1f77b4", 15000: "#f28e2b", 30000: "#2ca02c"}
    linestyles = {7500: "-", 15000: "--", 30000: "-."}
    for ax, (nfft, psd) in zip(axes, nfft_psds):
        ax.plot(
            psd.freqs,
            psd.db,
            color=colors[nfft],
            ls=linestyles[nfft],
            lw=1.65,
        )
        ax.axvspan(8, 13, color="#8ecae6", alpha=0.14, lw=0)
        ax.text(
            0.015,
            0.12,
            f"nfft={nfft}, Δf={fs / nfft:.4f} Hz",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#bcccdc", "alpha": 0.92},
        )
        ax.grid(True, alpha=0.24)
    axes[0].set_title("实验二(4-3)：同一30 s EEG补零后频率轴变密但分辨率不变")
    axes[1].set_ylabel("功率谱密度 (dB/Hz)")
    axes[-1].set_xlabel("频率 (Hz)")
    axes[-1].set_xlim(6, 16)
    axes[0].text(10.5, axes[0].get_ylim()[1], "α频段", ha="center", va="top", fontsize=9, color="#32566b")
    outputs["exp2_real_nfft_figure"] = save_figure(fig, "exp2_real_nfft_alpha.png")
    table_path = TABLE_DIR / "exp2_real_nfft_alpha_metrics.csv"
    pd.DataFrame(rows).to_csv(table_path, index=False, encoding="utf-8-sig")
    outputs["exp2_real_nfft_table"] = table_path

    np.savez_compressed(CACHE_DIR / "exp2_real_segments.npz", oz_59s=base, oz_30s=x30)
    return outputs


if __name__ == "__main__":
    run()
