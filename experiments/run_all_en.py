from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FS, OUTPUT_DIR, PROJECT_ROOT, RANDOM_SEED, ensure_output_dirs, next_pow_two
from src.data_io import load_eeg
from src.spectral_methods import ar_psd, fpe_values, periodogram_psd, welch_psd
from src.spectral_methods import autocorrelation_psd, direct_psd


FIGURE_EN_DIR = OUTPUT_DIR / "figures_en"


def setup() -> None:
    ensure_output_dirs()
    FIGURE_EN_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
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


def save(fig: plt.Figure, filename: str) -> Path:
    path = FIGURE_EN_DIR / filename
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def finish_psd(ax, xlim=(0, 40), ylabel="Power spectral density (dB/Hz)") -> None:
    ax.set_xlim(*xlim)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.24)


def add_alpha_band(ax, label="alpha band") -> None:
    ax.axvspan(8, 13, color="#8ecae6", alpha=0.15, lw=0)
    y0, y1 = ax.get_ylim()
    ax.text(10.5, y1, label, ha="center", va="top", fontsize=9, color="#32566b")


def run_all_en() -> dict[str, Path]:
    setup()
    outputs: dict[str, Path] = {}
    outputs.update(exp1_figures())
    outputs.update(exp2_figures())
    outputs.update(exp3_figures())
    outputs.update(exp4_figures())
    outputs["mechanism"] = mechanism_figure()
    return outputs


def exp1_figures() -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    rng = np.random.default_rng(RANDOM_SEED + 1)
    fs = FS
    t = np.arange(0.0, 4.0, 1.0 / fs)
    x = np.sin(2 * np.pi * 12.0 * t) + 0.6 * np.sin(2 * np.pi * 24.0 * t) + 0.8 * rng.standard_normal(len(t))
    nfft = 4096

    fig, ax = plt.subplots(figsize=(9.4, 3.4))
    ax.plot(t, x, color="#264653", lw=1.0)
    ax.set_title("Experiment 1: artificial composite signal in the time domain")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, 4)
    outputs["exp1_artificial_time"] = save(fig, "exp1_artificial_time_signal_en.png")

    psds = [
        direct_psd(x, fs, nfft=nfft, label="Direct FFT, rectangular window"),
        periodogram_psd(x, fs, window="hamming", nfft=nfft, label="Periodogram, Hamming window"),
        autocorrelation_psd(x, fs, nfft=nfft, label="Autocorrelation method"),
        welch_psd(x, fs, segment_s=1.0, overlap=0.5, window="hamming", nfft=nfft, label="Welch, 1 s / 50%"),
    ]
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    for psd in psds:
        ax.plot(psd.freqs, psd.db, label=psd.label)
    ax.axvline(12, color="#d62828", ls="--", lw=1.2, label="12 Hz")
    ax.axvline(24, color="#f77f00", ls="--", lw=1.2, label="24 Hz")
    ax.set_title("Experiment 1: four nonparametric PSD estimates for the artificial signal")
    finish_psd(ax, xlim=(0, 45))
    ax.legend(ncol=2, fontsize=9)
    outputs["exp1_artificial_methods"] = save(fig, "exp1_artificial_methods_en.png")

    eeg = load_eeg()
    x_eeg = eeg.segment("Oz", 58.84, 117.84)
    t_eeg = np.arange(len(x_eeg)) / eeg.fs + 58.84
    fig, ax = plt.subplots(figsize=(9.4, 3.4))
    ax.plot(t_eeg, x_eeg * 1e6, color="#264653", lw=0.65)
    ax.set_title("Experiment 1: Oz closed-eye EEG segment, 58.84-117.84 s")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Potential (uV)")
    ax.set_xlim(t_eeg[0], t_eeg[-1])
    outputs["exp1_real_time"] = save(fig, "exp1_real_eeg_time_segment_en.png")

    nfft = next_pow_two(2 * len(x_eeg))
    psds = [
        direct_psd(x_eeg, eeg.fs, nfft=nfft, label="Direct FFT, rectangular window"),
        periodogram_psd(x_eeg, eeg.fs, window="hamming", nfft=nfft, label="Periodogram, Hamming window"),
        autocorrelation_psd(x_eeg, eeg.fs, nfft=nfft, label="Autocorrelation method"),
        welch_psd(x_eeg, eeg.fs, segment_s=1.0, overlap=0.5, window="hamming", nfft=nfft, label="Welch, 1 s / 50%"),
    ]
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    for psd in psds:
        ax.plot(psd.freqs, psd.db, label=psd.label)
    add_alpha_band(ax)
    ax.set_title("Experiment 1: classical PSD estimates for Oz closed-eye EEG")
    finish_psd(ax, xlim=(0, 40))
    ax.legend(ncol=2, fontsize=9)
    outputs["exp1_real_methods"] = save(fig, "exp1_real_eeg_methods_en.png")
    return outputs


def exp2_figures() -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    fs = FS

    psds = []
    for duration in [1.0, 2.0, 4.0, 8.0]:
        t = np.arange(0.0, duration, 1.0 / fs)
        x = np.sin(2 * np.pi * 10.0 * t) + 0.8 * np.sin(2 * np.pi * 11.0 * t)
        psds.append(periodogram_psd(x, fs, window="boxcar", nfft=4096, label=f"{duration:g} s"))
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for psd in psds:
        ax.plot(psd.freqs, psd.db, label=psd.label)
    ax.axvline(10, color="#d62828", ls="--", lw=1.1)
    ax.axvline(11, color="#f77f00", ls="--", lw=1.1)
    ax.set_title("Experiment 2: record length and resolving 10 Hz / 11 Hz components")
    finish_psd(ax, xlim=(7, 14))
    ax.legend(title="Record length")
    outputs["exp2_length"] = save(fig, "exp2_length_resolution_en.png")

    t = np.arange(0.0, 1.8, 1.0 / fs)
    x = np.sin(2 * np.pi * 10.3 * t) + 0.15 * np.sin(2 * np.pi * 14.1 * t)
    windows = [("boxcar", "Rectangular"), ("hann", "Hann"), ("hamming", "Hamming"), ("blackman", "Blackman")]
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for window, label in windows:
        psd = periodogram_psd(x, fs, window=window, nfft=4096, label=label)
        ax.plot(psd.freqs, psd.db, label=label)
    ax.axvline(10.3, color="#d62828", ls="--", lw=1.1, label="10.3 Hz")
    ax.axvline(14.1, color="#2a9d8f", ls="--", lw=1.1, label="14.1 Hz")
    ax.set_title("Experiment 2: window leakage and weak-peak visibility")
    finish_psd(ax, xlim=(6, 18))
    ax.legend(ncol=2, fontsize=9)
    outputs["exp2_window"] = save(fig, "exp2_window_leakage_en.png")

    t = np.arange(0.0, 1.0, 1.0 / fs)
    x = np.sin(2 * np.pi * 10.0 * t) + 0.8 * np.sin(2 * np.pi * 10.4 * t)
    nffts = [256, 2048, 8192]
    fig, axes = plt.subplots(len(nffts), 1, figsize=(8.8, 7.6), sharex=True)
    for ax, nfft in zip(axes, nffts):
        psd = periodogram_psd(x, fs, window="boxcar", nfft=nfft, label=f"nfft={nfft}")
        ax.plot(psd.freqs, psd.db, color="#264653")
        ax.axvline(10.0, color="#d62828", ls="--", lw=1.0)
        ax.axvline(10.4, color="#f77f00", ls="--", lw=1.0)
        ax.set_title(f"nfft={nfft}, grid spacing={fs / nfft:.4f} Hz", fontsize=10)
        finish_psd(ax, xlim=(8.5, 12.2))
    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle("Experiment 2: zero padding densifies the frequency grid but not physical resolution", y=0.995)
    outputs["exp2_nfft"] = save(fig, "exp2_nfft_subplots_en.png")

    eeg = load_eeg()
    base = eeg.segment("Oz", 58.84, 117.84)
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for duration in [10.0, 30.0, 59.0]:
        x = base[: int(round(duration * eeg.fs))]
        nfft = next_pow_two(len(x))
        psd = periodogram_psd(x, eeg.fs, window="hamming", nfft=nfft, label=f"{duration:g} s, nfft={nfft}")
        ax.plot(psd.freqs, psd.db, label=psd.label)
    add_alpha_band(ax)
    ax.set_title("Experiment 2: EEG alpha peak under different record lengths")
    finish_psd(ax, xlim=(2, 30))
    ax.legend(fontsize=9)
    outputs["exp2_real_length"] = save(fig, "exp2_real_length_alpha_en.png")

    x30 = base[: int(round(30.0 * eeg.fs))]
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for window, label in [("boxcar", "Rectangular"), ("hamming", "Hamming"), ("blackman", "Blackman")]:
        psd = periodogram_psd(x30, eeg.fs, window=window, nfft=8192, label=label)
        ax.plot(psd.freqs, psd.db, label=label)
    add_alpha_band(ax)
    ax.set_title("Experiment 2: EEG alpha peak under different windows")
    finish_psd(ax, xlim=(2, 30))
    ax.legend(fontsize=9)
    outputs["exp2_real_window"] = save(fig, "exp2_real_windows_alpha_en.png")

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for nfft in [7500, 15000, 30000]:
        psd = periodogram_psd(x30, eeg.fs, window="hamming", nfft=nfft, label=f"nfft={nfft}, df={eeg.fs / nfft:.4f} Hz")
        ax.plot(psd.freqs, psd.db, label=psd.label)
    add_alpha_band(ax)
    ax.set_title("Experiment 2: same 30 s EEG segment with different NFFT values")
    finish_psd(ax, xlim=(6, 16))
    ax.legend(fontsize=9)
    outputs["exp2_real_nfft"] = save(fig, "exp2_real_nfft_alpha_en.png")
    return outputs


def exp3_figures() -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    fs = FS
    rng = np.random.default_rng(RANDOM_SEED + 3)
    t = np.arange(0.0, 2.0, 1.0 / fs)
    x = np.sin(2 * np.pi * 10.0 * t) + 0.8 * np.sin(2 * np.pi * 11.0 * t) + rng.standard_normal(len(t))
    nfft = 4096
    orders = [4, 16, 32, 128]
    welch = welch_psd(x, fs, segment_s=1.0, overlap=0.5, window="hamming", nfft=nfft, label="Welch, 1 s / 50%")
    ar_psds = [ar_psd(x, fs, order=p, nfft=nfft, label=f"AR-YW p={p}") for p in orders]

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.plot(welch.freqs, welch.db, label=welch.label, color="#111827", lw=2.1)
    for psd in ar_psds:
        ax.plot(psd.freqs, psd.db, label=psd.label)
    ax.axvline(10, color="#d62828", ls="--", lw=1.0)
    ax.axvline(11, color="#f77f00", ls="--", lw=1.0)
    ax.set_title("Experiment 3: Welch and AR spectra for the artificial signal")
    finish_psd(ax, xlim=(6, 18))
    ax.legend(ncol=2, fontsize=9)
    outputs["exp3_artificial"] = save(fig, "exp3_artificial_welch_ar_en.png")

    fpe_df = pd.DataFrame(fpe_values(x, range(2, 31)))
    p_star = int(fpe_df.loc[fpe_df["fpe"].idxmin(), "order"])
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(fpe_df["order"], fpe_df["fpe"] / fpe_df["fpe"].min(), marker="o", ms=3.8, color="#264653")
    ax.axvline(p_star, color="#d62828", ls="--", lw=1.2, label=f"minimum FPE, p*={p_star}")
    ax.set_title("Experiment 3: artificial signal AR order-relative FPE curve")
    ax.set_xlabel("AR order p")
    ax.set_ylabel("Relative FPE (minimum=1)")
    ax.legend()
    outputs["exp3_artificial_fpe"] = save(fig, "exp3_artificial_fpe_en.png")

    compare_orders = sorted(set([*orders, p_star]))
    compare_psds = [ar_psd(x, fs, order=p, nfft=nfft, label=f"AR-YW p={p}" + (" (FPE)" if p == p_star else "")) for p in compare_orders]
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for psd in compare_psds:
        ax.plot(psd.freqs, psd.db, label=psd.label)
    ax.axvline(10, color="#d62828", ls="--", lw=1.0)
    ax.axvline(11, color="#f77f00", ls="--", lw=1.0)
    ax.set_title("Experiment 3: FPE-selected AR order versus representative orders")
    finish_psd(ax, xlim=(6, 18))
    ax.legend(ncol=2, fontsize=9)
    outputs["exp3_artificial_pstar"] = save(fig, "exp3_artificial_ar_pstar_en.png")

    eeg = load_eeg()
    x_eeg = eeg.segment("Oz", 126.61, 134.61)
    nfft = 8192
    fpe_df = pd.DataFrame(fpe_values(x_eeg, range(2, 81)))
    p_star = int(fpe_df.loc[fpe_df["fpe"].idxmin(), "order"])
    orders = [4, 16, 32, 64]
    welch = welch_psd(x_eeg, eeg.fs, segment_s=2.0, overlap=0.5, window="hamming", nfft=nfft, label="Welch, 2 s / 50%")
    ar_psds = [ar_psd(x_eeg, eeg.fs, order=p, nfft=nfft, label=f"AR-YW p={p}" + (" (FPE)" if p == p_star else "")) for p in sorted(set([*orders, p_star]))]
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    ax.plot(welch.freqs, welch.db, label=welch.label, color="#111827", lw=2.2)
    for psd in ar_psds:
        ax.plot(psd.freqs, psd.db, label=psd.label)
    add_alpha_band(ax)
    ax.set_title("Experiment 3: Welch and AR spectra for the real Oz EEG segment")
    finish_psd(ax, xlim=(2, 40))
    ax.legend(ncol=2, fontsize=8.6)
    outputs["exp3_real"] = save(fig, "exp3_real_welch_ar_en.png")

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.plot(welch.freqs, welch.db, label=welch.label, color="#111827", lw=2.2)
    for psd in ar_psds:
        ax.plot(psd.freqs, psd.db, label=psd.label)
    add_alpha_band(ax)
    ax.set_title("Experiment 3: alpha-band zoom for the real Oz EEG segment")
    finish_psd(ax, xlim=(7, 14))
    ax.legend(ncol=2, fontsize=8.6)
    outputs["exp3_real_zoom"] = save(fig, "exp3_real_welch_ar_alpha_zoom_en.png")

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(fpe_df["order"], fpe_df["fpe"] / fpe_df["fpe"].min(), marker="o", ms=3.2, color="#264653")
    ax.axvline(p_star, color="#d62828", ls="--", lw=1.2, label=f"minimum FPE, p*={p_star}")
    ax.set_title("Experiment 3: real EEG AR order-relative FPE curve")
    ax.set_xlabel("AR order p")
    ax.set_ylabel("Relative FPE (minimum=1)")
    ax.legend()
    outputs["exp3_real_fpe"] = save(fig, "exp3_real_fpe_en.png")
    return outputs


def exp4_figures() -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    fs = FS
    rng = np.random.default_rng(RANDOM_SEED + 4)
    t = np.arange(0.0, 8.0, 1.0 / fs)
    signals = np.asarray([np.sin(2 * np.pi * 10.0 * t) + rng.standard_normal(len(t)) for _ in range(30)])
    nfft = 4096
    periodograms = [periodogram_psd(x, fs, window="hamming", nfft=nfft, label=f"realization {i+1}") for i, x in enumerate(signals)]
    welches = [welch_psd(x, fs, segment_s=2.0, overlap=0.5, window="hamming", nfft=nfft, label=f"realization {i+1}") for i, x in enumerate(signals)]
    outputs["exp4_art_periodogram"] = plot_many_with_mean(periodograms, "Experiment 4: 30 periodograms from the same random process", "exp4_artificial_periodogram_30_en.png")
    outputs["exp4_art_welch"] = plot_many_with_mean(welches, "Experiment 4: 30 Welch estimates from the same random process", "exp4_artificial_welch_30_en.png")
    outputs["exp4_art_std"] = plot_std_comparison(
        [("Single periodogram", periodograms), ("Welch, 2 s / 50%", welches)],
        "Experiment 4: cross-realization PSD standard deviation",
        "exp4_artificial_std_comparison_en.png",
    )

    mean_psds = []
    for segment_s in [1.0, 2.0, 4.0]:
        psds = [welch_psd(x, fs, segment_s=segment_s, overlap=0.5, window="hamming", nfft=nfft, label=f"{segment_s:g} s") for x in signals]
        stack = np.vstack([p.pxx for p in psds])
        mean_psds.append((segment_s, psds[0].freqs, 10 * np.log10(np.maximum(np.mean(stack, axis=0), np.finfo(float).tiny))))
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for segment_s, freqs, mean_db in mean_psds:
        ax.plot(freqs, mean_db, label=f"Welch segment {segment_s:g} s")
    ax.axvline(10, color="#d62828", ls="--", lw=1.0)
    ax.set_title("Experiment 4: Welch segment-length trade-off")
    finish_psd(ax, xlim=(4, 24))
    ax.legend()
    outputs["exp4_segment"] = save(fig, "exp4_welch_segment_lengths_en.png")

    eeg = load_eeg()
    x = eeg.segment("Oz", 58.84, 112.84)
    win_n = int(round(6.0 * eeg.fs))
    windows = np.vstack([x[i * win_n : (i + 1) * win_n] for i in range(9)])
    periodograms = [periodogram_psd(win, eeg.fs, window="hamming", nfft=nfft, label=f"window {i+1}") for i, win in enumerate(windows)]
    welches = [welch_psd(win, eeg.fs, segment_s=2.0, overlap=0.5, window="hamming", nfft=nfft, label=f"window {i+1}") for i, win in enumerate(windows)]
    outputs["exp4_real_periodogram"] = plot_many_with_mean(periodograms, "Experiment 4: real EEG periodograms across 6 s windows", "exp4_real_periodogram_windows_en.png", alpha_band=True)
    outputs["exp4_real_welch"] = plot_many_with_mean(welches, "Experiment 4: real EEG Welch estimates across 6 s windows", "exp4_real_welch_windows_en.png", alpha_band=True)
    outputs["exp4_real_std"] = plot_std_comparison(
        [("Single periodogram", periodograms), ("Welch, 2 s / 50%", welches)],
        "Experiment 4: real EEG cross-window PSD standard deviation",
        "exp4_real_std_comparison_en.png",
        alpha_band=True,
    )
    return outputs


def plot_many_with_mean(psds, title: str, filename: str, alpha_band: bool = False) -> Path:
    freqs = psds[0].freqs
    pxx_stack = np.vstack([p.pxx for p in psds])
    db_stack = 10.0 * np.log10(np.maximum(pxx_stack, np.finfo(float).tiny))
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for row in db_stack:
        ax.plot(freqs, row, color="#7aa6c2", alpha=0.28, lw=1.0)
    mean_db = 10.0 * np.log10(np.maximum(np.mean(pxx_stack, axis=0), np.finfo(float).tiny))
    ax.plot(freqs, mean_db, color="#d62828", lw=2.4, label="mean spectrum")
    if alpha_band:
        add_alpha_band(ax)
    else:
        ax.axvline(10, color="#111827", ls="--", lw=1.0, label="10 Hz")
    ax.set_title(title)
    finish_psd(ax, xlim=(2, 35))
    ax.legend()
    return save(fig, filename)


def plot_std_comparison(named_psds, title: str, filename: str, alpha_band: bool = False) -> Path:
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    freqs = named_psds[0][1][0].freqs
    for label, psds in named_psds:
        stack = np.vstack([psd.db for psd in psds])
        ax.plot(freqs, np.std(stack, axis=0, ddof=1), label=label)
    if alpha_band:
        add_alpha_band(ax)
    else:
        ax.axvline(10, color="#111827", ls="--", lw=1.0, label="10 Hz")
    ax.set_title(title)
    ax.set_xlim(2, 35)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD standard deviation (dB)")
    ax.grid(True, alpha=0.24)
    ax.legend()
    return save(fig, filename)


def mechanism_figure() -> Path:
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(14, 8.0))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.94, "Spectral Estimation Workflow", ha="center", va="center", fontsize=20, weight="bold", color="#102a43")
    ax.text(
        0.5,
        0.895,
        "Finite observations, estimator assumptions, and display choices jointly shape spectral interpretation",
        ha="center",
        va="center",
        fontsize=11,
        color="#334e68",
    )

    pipeline = [
        ("Finite data", "EEG segment\nor simulated signal", 0.035, "#d8f3dc"),
        ("Analysis choices", "record length\nwindow type\nNFFT grid", 0.235, "#b7e4c7"),
        ("Estimator", "periodogram\nWelch\nYule-Walker AR", 0.435, "#95d5b2"),
        ("Measured features", "peak frequency\nbandwidth\nprominence", 0.635, "#74c69d"),
        ("Conclusion", "bias-variance\nresolution\nphysiology", 0.835, "#52b788"),
    ]
    y, w, h = 0.62, 0.14, 0.17
    for title, body, x, color in pipeline:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.018,rounding_size=0.018", fc=color, ec="#1b4332", lw=1.6)
        ax.add_patch(patch)
        ax.text(
            x + w / 2,
            y + h * 0.66,
            title,
            ha="center",
            va="center",
            fontsize=12,
            weight="bold",
            color="#102a43",
        )
        ax.text(x + w / 2, y + h * 0.36, body, ha="center", va="center", fontsize=10, color="#102a43", linespacing=1.25)
    for _, _, x, _ in pipeline[:-1]:
        ax.annotate("", xy=(x + w + 0.028, y + h / 2), xytext=(x + w + 0.006, y + h / 2), arrowprops=dict(arrowstyle="->", lw=2.0, color="#1b4332"))

    ax.text(0.5, 0.51, "Core trade-offs", ha="center", va="center", fontsize=15, weight="bold", color="#102a43")
    tradeoffs = [
        ("Resolution", "Longer records reduce\nRayleigh spacing and\nnarrow the main lobe.", 0.06, "#e9f5db"),
        ("Leakage", "Low-sidelobe windows\nsuppress leakage but\nusually widen peaks.", 0.37, "#d8f3dc"),
        ("Stability", "Welch averaging lowers\nvariance, while AR order\ncan sharpen or overfit.", 0.68, "#c7f9cc"),
    ]
    y2, w2, h2 = 0.22, 0.25, 0.20
    for title, body, x, color in tradeoffs:
        patch = FancyBboxPatch((x, y2), w2, h2, boxstyle="round,pad=0.02,rounding_size=0.02", fc=color, ec="#40916c", lw=1.4)
        ax.add_patch(patch)
        ax.text(x + w2 / 2, y2 + h2 * 0.70, title, ha="center", va="center", fontsize=12, weight="bold", color="#1b4332")
        ax.text(x + w2 / 2, y2 + h2 * 0.38, body, ha="center", va="center", fontsize=10, color="#1b4332", linespacing=1.3)
    ax.text(
        0.5,
        0.10,
        "Zero padding only interpolates the frequency grid; it does not create new information or physical resolution.",
        ha="center",
        va="center",
        fontsize=11,
        color="#334e68",
        bbox=dict(boxstyle="round,pad=0.35", fc="#f8f9fa", ec="#bcccdc", lw=1.0),
    )
    return save(fig, "spectral_pipeline_mechanism_en.png")


if __name__ == "__main__":
    run_all_en()
