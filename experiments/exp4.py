from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FS, RANDOM_SEED, TABLE_DIR, ensure_output_dirs
from src.data_io import load_eeg
from src.plotting import add_band, finish_psd_axes, save_figure, setup_matplotlib
from src.spectral_methods import alpha_metrics, band_mean_db, minus_3db_width, periodogram_psd, pointwise_std_db, value_at, welch_psd


def run() -> dict[str, Path]:
    ensure_output_dirs()
    setup_matplotlib()
    outputs: dict[str, Path] = {}
    signals = _generate_artificial_signals()
    outputs.update(_same_process_realizations(signals))
    outputs.update(_welch_segment_length_tradeoff(signals))
    outputs.update(_real_eeg_windows())
    return outputs


def _generate_artificial_signals(n_realizations: int = 30) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_SEED + 4)
    t = np.arange(0.0, 8.0, 1.0 / FS)
    signals = []
    for _ in range(n_realizations):
        signals.append(np.sin(2 * np.pi * 10.0 * t) + rng.standard_normal(len(t)))
    return np.asarray(signals)


def _same_process_realizations(signals: np.ndarray) -> dict[str, Path]:
    fs = FS
    nfft = 4096
    periodograms = [periodogram_psd(x, fs, window="hamming", nfft=nfft, label=f"实现{i+1}") for i, x in enumerate(signals)]
    welches = [
        welch_psd(x, fs, segment_s=2.0, overlap=0.5, window="hamming", nfft=nfft, label=f"实现{i+1}")
        for i, x in enumerate(signals)
    ]

    periodogram_stack = np.vstack([p.pxx for p in periodograms])
    welch_stack = np.vstack([p.pxx for p in welches])
    freqs = periodograms[0].freqs

    periodogram_fig = _plot_many_with_mean(
        freqs,
        periodogram_stack,
        "实验四(1)：30个同一随机过程实现的单次周期图",
        "exp4_artificial_periodogram_30.png",
    )
    welch_fig = _plot_many_with_mean(
        freqs,
        welch_stack,
        "实验四(1)：30个同一随机过程实现的Welch谱",
        "exp4_artificial_welch_30.png",
    )
    std_fig = _plot_std_comparison(
        freqs,
        [("单次周期图", periodograms), ("Welch(2s/50%)", welches)],
        "实验四(1)：同一随机过程跨实现PSD标准差",
        "exp4_artificial_std_comparison.png",
    )

    rows = []
    for method, psds in [("单次周期图", periodograms), ("Welch(2s/50%)", welches)]:
        per_rows = []
        for psd in psds:
            db = psd.db
            alpha = alpha_metrics(psd.freqs, db)
            per_rows.append(
                {
                    "power_10hz_db_hz": value_at(psd.freqs, db, 10.0),
                    "alpha_peak_hz": alpha["alpha_peak_hz"],
                    "alpha_prominence_db": alpha["alpha_prominence_db"],
                    "background_15_30_mean_db_hz": band_mean_db(psd.freqs, db, (15.0, 30.0)),
                }
            )
        frame = pd.DataFrame(per_rows)
        rows.append(
            {
                "method": method,
                "power_10hz_mean_db_hz": frame["power_10hz_db_hz"].mean(),
                "power_10hz_std_db": frame["power_10hz_db_hz"].std(ddof=1),
                "alpha_peak_hz_mean": frame["alpha_peak_hz"].mean(),
                "alpha_peak_hz_std": frame["alpha_peak_hz"].std(ddof=1),
                "alpha_prominence_mean_db": frame["alpha_prominence_db"].mean(),
                "alpha_prominence_std_db": frame["alpha_prominence_db"].std(ddof=1),
                "background_15_30_mean_std_db": frame["background_15_30_mean_db_hz"].std(ddof=1),
                "pointwise_std_8_13_mean_db": pointwise_std_db(psds, (8.0, 13.0)),
                "pointwise_std_15_30_mean_db": pointwise_std_db(psds, (15.0, 30.0)),
            }
        )
    table_path = TABLE_DIR / "exp4_artificial_realization_stability_metrics.csv"
    pd.DataFrame(rows).to_csv(table_path, index=False, encoding="utf-8-sig")
    return {
        "exp4_artificial_periodogram_figure": periodogram_fig,
        "exp4_artificial_welch_figure": welch_fig,
        "exp4_artificial_std_figure": std_fig,
        "exp4_artificial_stability_table": table_path,
    }


def _welch_segment_length_tradeoff(signals: np.ndarray) -> dict[str, Path]:
    fs = FS
    nfft = 4096
    segment_lengths = [1.0, 2.0, 4.0]
    rows = []
    mean_psds = []
    freqs = None
    for segment_s in segment_lengths:
        psds = [
            welch_psd(x, fs, segment_s=segment_s, overlap=0.5, window="hamming", nfft=nfft, label=f"{segment_s:g} s")
            for x in signals
        ]
        freqs = psds[0].freqs
        db_stack = np.vstack([p.db for p in psds])
        pxx_stack = np.vstack([p.pxx for p in psds])
        mean_pxx = np.mean(pxx_stack, axis=0)
        mean_db = 10.0 * np.log10(np.maximum(mean_pxx, np.finfo(float).tiny))
        mean_psds.append((segment_s, mean_db))
        per_alpha = [alpha_metrics(p.freqs, p.db) for p in psds]
        alpha_frame = pd.DataFrame(per_alpha)
        rows.append(
            {
                "segment_length_s": segment_s,
                "power_10hz_std_db": float(np.std([value_at(p.freqs, p.db, 10.0) for p in psds], ddof=1)),
                "alpha_peak_hz_std": float(alpha_frame["alpha_peak_hz"].std(ddof=1)),
                "alpha_prominence_mean_db": float(alpha_frame["alpha_prominence_db"].mean()),
                "alpha_prominence_std_db": float(alpha_frame["alpha_prominence_db"].std(ddof=1)),
                "pointwise_std_8_13_mean_db": pointwise_std_db(psds, (8.0, 13.0)),
                "pointwise_std_15_30_mean_db": pointwise_std_db(psds, (15.0, 30.0)),
                "mean_spectrum_alpha_minus_3db_width_hz": minus_3db_width(freqs, mean_db, (8.0, 13.0), (6.0, 16.0)),
            }
        )

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    assert freqs is not None
    for segment_s, mean_db in mean_psds:
        ax.plot(freqs, mean_db, label=f"Welch分段 {segment_s:g} s")
    ax.axvline(10, color="#d62828", ls="--", lw=1.0)
    ax.set_title("实验四(2)：Welch分段长度的统计稳定性与频率分辨率折中")
    finish_psd_axes(ax, xlim=(4, 24))
    ax.legend()
    fig_path = save_figure(fig, "exp4_welch_segment_lengths.png")
    table_path = TABLE_DIR / "exp4_welch_segment_length_metrics.csv"
    pd.DataFrame(rows).to_csv(table_path, index=False, encoding="utf-8-sig")
    return {"exp4_segment_figure": fig_path, "exp4_segment_table": table_path}


def _real_eeg_windows() -> dict[str, Path]:
    eeg = load_eeg()
    fs = eeg.fs
    x = eeg.segment("Oz", 58.84, 112.84)
    n_win = 9
    win_n = int(round(6.0 * fs))
    windows = np.vstack([x[i * win_n : (i + 1) * win_n] for i in range(n_win)])
    nfft = 4096
    periodograms = [
        periodogram_psd(win, fs, window="hamming", nfft=nfft, label=f"窗{i+1}") for i, win in enumerate(windows)
    ]
    welches = [
        welch_psd(win, fs, segment_s=2.0, overlap=0.5, window="hamming", nfft=nfft, label=f"窗{i+1}")
        for i, win in enumerate(windows)
    ]

    freqs = periodograms[0].freqs
    periodogram_stack = np.vstack([p.pxx for p in periodograms])
    welch_stack = np.vstack([p.pxx for p in welches])

    periodogram_fig = _plot_many_with_mean(
        freqs,
        periodogram_stack,
        "实验四(3)：真实Oz 9个6 s时间窗的单次周期图",
        "exp4_real_periodogram_windows.png",
        alpha_band=True,
    )
    welch_fig = _plot_many_with_mean(
        freqs,
        welch_stack,
        "实验四(3)：真实Oz 9个6 s时间窗的Welch谱",
        "exp4_real_welch_windows.png",
        alpha_band=True,
    )
    std_fig = _plot_std_comparison(
        freqs,
        [("单次周期图", periodograms), ("Welch(2s/50%)", welches)],
        "实验四(3)：真实Oz跨时间窗PSD标准差",
        "exp4_real_std_comparison.png",
        alpha_band=True,
    )

    rows = []
    for method, psds, stack in [("单次周期图", periodograms, periodogram_stack), ("Welch(2s/50%)", welches, welch_stack)]:
        alpha_frame = pd.DataFrame([alpha_metrics(p.freqs, p.db) for p in psds])
        mean_db = 10 * np.log10(np.maximum(np.mean(stack, axis=0), np.finfo(float).tiny))
        rows.append(
            {
                "method": method,
                "alpha_peak_hz_mean": float(alpha_frame["alpha_peak_hz"].mean()),
                "alpha_peak_hz_std": float(alpha_frame["alpha_peak_hz"].std(ddof=1)),
                "alpha_prominence_mean_db": float(alpha_frame["alpha_prominence_db"].mean()),
                "alpha_prominence_std_db": float(alpha_frame["alpha_prominence_db"].std(ddof=1)),
                "pointwise_std_8_13_mean_db": pointwise_std_db(psds, (8.0, 13.0)),
                "pointwise_std_15_30_mean_db": pointwise_std_db(psds, (15.0, 30.0)),
                "mean_spectrum_alpha_minus_3db_width_hz": minus_3db_width(freqs, mean_db, (8.0, 13.0), (6.0, 16.0)),
            }
        )
    table_path = TABLE_DIR / "exp4_real_window_stability_metrics.csv"
    pd.DataFrame(rows).to_csv(table_path, index=False, encoding="utf-8-sig")
    return {
        "exp4_real_periodogram_figure": periodogram_fig,
        "exp4_real_welch_figure": welch_fig,
        "exp4_real_std_figure": std_fig,
        "exp4_real_stability_table": table_path,
    }


def _plot_many_with_mean(
    freqs: np.ndarray,
    pxx_stack: np.ndarray,
    title: str,
    filename: str,
    alpha_band: bool = False,
) -> Path:
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    db_stack = 10.0 * np.log10(np.maximum(pxx_stack, np.finfo(float).tiny))
    for row in db_stack:
        ax.plot(freqs, row, color="#7aa6c2", alpha=0.28, lw=1.0)
    mean_db = 10.0 * np.log10(np.maximum(np.mean(pxx_stack, axis=0), np.finfo(float).tiny))
    ax.plot(freqs, mean_db, color="#d62828", lw=2.4, label="均值谱")
    if alpha_band:
        add_band(ax, 8, 13, "α频段")
    else:
        ax.axvline(10, color="#111827", ls="--", lw=1.0, label="10 Hz")
    ax.set_title(title)
    finish_psd_axes(ax, xlim=(2, 35))
    ax.legend()
    return save_figure(fig, filename)


def _plot_std_comparison(
    freqs: np.ndarray,
    named_psds: list[tuple[str, list]],
    title: str,
    filename: str,
    alpha_band: bool = False,
) -> Path:
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for label, psds in named_psds:
        stack = np.vstack([psd.db for psd in psds])
        ax.plot(freqs, np.std(stack, axis=0, ddof=1), label=label)
    if alpha_band:
        add_band(ax, 8, 13, "α频段")
    else:
        ax.axvline(10, color="#111827", ls="--", lw=1.0, label="10 Hz")
    ax.set_title(title)
    ax.set_xlim(2, 35)
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("跨实现/时间窗标准差 (dB)")
    ax.grid(True, alpha=0.24)
    ax.legend()
    return save_figure(fig, filename)


if __name__ == "__main__":
    run()
