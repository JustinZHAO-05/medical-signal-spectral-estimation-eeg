from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FS, RANDOM_SEED, TABLE_DIR, ensure_output_dirs
from src.data_io import load_eeg
from src.plotting import add_band, finish_psd_axes, save_figure, setup_matplotlib
from src.spectral_methods import (
    alpha_metrics,
    ar_psd,
    count_local_maxima,
    fpe_values,
    minus_3db_width,
    peak_resolution_metrics,
    welch_psd,
)


def run(*, synthetic_only: bool = False) -> dict[str, Path]:
    ensure_output_dirs()
    setup_matplotlib()
    outputs: dict[str, Path] = {}
    outputs.update(_artificial_welch_ar())
    if not synthetic_only:
        outputs.update(_real_eeg_welch_ar())
    return outputs


def _artificial_signal() -> np.ndarray:
    rng = np.random.default_rng(RANDOM_SEED + 3)
    t = np.arange(0.0, 2.0, 1.0 / FS)
    return np.sin(2 * np.pi * 10.0 * t) + 0.8 * np.sin(2 * np.pi * 11.0 * t) + rng.standard_normal(len(t))


def _artificial_welch_ar() -> dict[str, Path]:
    fs = FS
    x = _artificial_signal()
    nfft = 4096
    orders = [4, 16, 32, 128]
    welch = welch_psd(x, fs, segment_s=1.0, overlap=0.5, window="hamming", nfft=nfft, label="Welch(1s/50%)")
    ar_psds = [ar_psd(x, fs, order=p, nfft=nfft, label=f"AR-YW p={p}") for p in orders]

    rows = []
    risk_rows = []
    for psd in [welch, *ar_psds]:
        row = {"method": psd.label}
        row.update(peak_resolution_metrics(psd.freqs, psd.db, (9.0, 12.0), prominence_db=2.0, min_distance_hz=0.25))
        rows.append(row)
        risk_rows.append(
            {
                "method": psd.label,
                "local_peak_count_6_18_hz": count_local_maxima(psd.freqs, psd.db, (6.0, 18.0), prominence_db=2.0, min_distance_hz=0.25),
                "local_peak_count_12_40_hz": count_local_maxima(psd.freqs, psd.db, (12.0, 40.0), prominence_db=2.0, min_distance_hz=0.5),
                "minus_3db_width_near_10hz": minus_3db_width(psd.freqs, psd.db, (9.0, 11.0), (6.0, 14.0)),
            }
        )

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.plot(welch.freqs, welch.db, label=welch.label, color="#111827", lw=2.1)
    for psd in ar_psds:
        ax.plot(psd.freqs, psd.db, label=psd.label)
    ax.axvline(10, color="#d62828", ls="--", lw=1.0)
    ax.axvline(11, color="#f77f00", ls="--", lw=1.0)
    ax.set_title("实验三(1)：Welch谱与不同阶数AR参数谱的人工信号比较")
    finish_psd_axes(ax, xlim=(6, 18))
    ax.legend(ncol=2, fontsize=9)
    fig_path = save_figure(fig, "exp3_artificial_welch_ar.png")

    metric_path = TABLE_DIR / "exp3_artificial_welch_ar_metrics.csv"
    pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna("未分辨").to_csv(
        metric_path, index=False, encoding="utf-8-sig"
    )
    risk_path = TABLE_DIR / "exp3_artificial_ar_peak_risk_metrics.csv"
    pd.DataFrame(risk_rows).replace([np.inf, -np.inf], np.nan).fillna("未分辨").to_csv(
        risk_path, index=False, encoding="utf-8-sig"
    )

    fpe_rows = fpe_values(x, range(2, 31))
    fpe_df = pd.DataFrame(fpe_rows)
    p_star = int(fpe_df.loc[fpe_df["fpe"].idxmin(), "order"])
    fpe_df.to_csv(TABLE_DIR / "exp3_artificial_fpe_values.csv", index=False, encoding="utf-8-sig", float_format="%.15f")

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(fpe_df["order"], fpe_df["fpe"] / fpe_df["fpe"].min(), marker="o", ms=3.8, color="#264653")
    ax.axvline(p_star, color="#d62828", ls="--", lw=1.2, label=f"FPE最小 p*={p_star}")
    ax.set_title("实验三(2)：人工信号AR阶数-相对FPE曲线")
    ax.set_xlabel("AR阶数 p")
    ax.set_ylabel("相对FPE (最小值=1)")
    ax.legend()
    fpe_fig_path = save_figure(fig, "exp3_artificial_fpe.png")

    compare_orders = sorted(set([*orders, p_star]))
    compare_psds = [ar_psd(x, fs, order=p, nfft=nfft, label=f"AR-YW p={p}" + (" (FPE)" if p == p_star else "")) for p in compare_orders]
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for psd in compare_psds:
        ax.plot(psd.freqs, psd.db, label=psd.label)
    ax.axvline(10, color="#d62828", ls="--", lw=1.0)
    ax.axvline(11, color="#f77f00", ls="--", lw=1.0)
    ax.set_title("实验三(2)：FPE推荐阶数与代表性AR谱的视觉比较")
    finish_psd_axes(ax, xlim=(6, 18))
    ax.legend(ncol=2, fontsize=9)
    pstar_fig_path = save_figure(fig, "exp3_artificial_ar_pstar.png")

    summary_path = TABLE_DIR / "exp3_artificial_fpe_summary.csv"
    pd.DataFrame([{"p_star": p_star, "min_fpe": float(fpe_df["fpe"].min()), "algorithm": "Yule-Walker"}]).to_csv(
        summary_path, index=False, encoding="utf-8-sig", float_format="%.15f"
    )
    return {
        "exp3_artificial_figure": fig_path,
        "exp3_artificial_table": metric_path,
        "exp3_artificial_risk_table": risk_path,
        "exp3_artificial_fpe_figure": fpe_fig_path,
        "exp3_artificial_pstar_figure": pstar_fig_path,
        "exp3_artificial_fpe_summary": summary_path,
    }


def _real_eeg_welch_ar() -> dict[str, Path]:
    eeg = load_eeg()
    fs = eeg.fs
    x = eeg.segment("Oz", 126.61, 134.61)
    nfft = 8192
    orders = [4, 16, 32, 64]

    fpe_df = pd.DataFrame(fpe_values(x, range(2, 81)))
    p_star = int(fpe_df.loc[fpe_df["fpe"].idxmin(), "order"])
    fpe_path = TABLE_DIR / "exp3_real_fpe_values.csv"
    fpe_df.to_csv(fpe_path, index=False, encoding="utf-8-sig", float_format="%.15f")

    welch = welch_psd(x, fs, segment_s=2.0, overlap=0.5, window="hamming", nfft=nfft, label="Welch(2s/50%)")
    compare_orders = sorted(set([*orders, p_star]))
    ar_psds = [
        ar_psd(x, fs, order=p, nfft=nfft, label=f"AR-YW p={p}" + (" (FPE)" if p == p_star else ""))
        for p in compare_orders
    ]

    rows = []
    risk_rows = []
    for psd in [welch, *ar_psds]:
        row = {"method": psd.label, "p_star": p_star, "algorithm": "Yule-Walker"}
        row.update(alpha_metrics(psd.freqs, psd.db))
        rows.append(row)
        risk_rows.append(
            {
                "method": psd.label,
                "local_peak_count_alpha_8_13_hz": count_local_maxima(psd.freqs, psd.db, (8.0, 13.0), prominence_db=1.0, min_distance_hz=0.25),
                "local_peak_count_13_40_hz": count_local_maxima(psd.freqs, psd.db, (13.0, 40.0), prominence_db=1.5, min_distance_hz=0.5),
                "alpha_minus_3db_width_hz": alpha_metrics(psd.freqs, psd.db)["alpha_minus_3db_width_hz"],
            }
        )
    metric_path = TABLE_DIR / "exp3_real_welch_ar_alpha_metrics.csv"
    pd.DataFrame(rows).to_csv(metric_path, index=False, encoding="utf-8-sig")
    risk_path = TABLE_DIR / "exp3_real_ar_peak_risk_metrics.csv"
    pd.DataFrame(risk_rows).to_csv(risk_path, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    ax.plot(welch.freqs, welch.db, label=welch.label, color="#111827", lw=2.2)
    for psd in ar_psds:
        ax.plot(psd.freqs, psd.db, label=psd.label)
    add_band(ax, 8, 13, "α频段")
    ax.set_title("实验三(3)：真实Oz 8 s闭眼EEG的Welch谱与AR谱比较")
    finish_psd_axes(ax, xlim=(2, 40))
    ax.legend(ncol=2, fontsize=8.6)
    spec_fig_path = save_figure(fig, "exp3_real_welch_ar.png")

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.plot(welch.freqs, welch.db, label=welch.label, color="#111827", lw=2.2)
    for psd in ar_psds:
        ax.plot(psd.freqs, psd.db, label=psd.label)
    add_band(ax, 8, 13, "α频段")
    ax.set_title("实验三(3)：真实Oz 8 s EEG的α频段局部放大")
    finish_psd_axes(ax, xlim=(7, 14))
    ax.legend(ncol=2, fontsize=8.6)
    zoom_fig_path = save_figure(fig, "exp3_real_welch_ar_alpha_zoom.png")

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(fpe_df["order"], fpe_df["fpe"] / fpe_df["fpe"].min(), marker="o", ms=3.2, color="#264653")
    ax.axvline(p_star, color="#d62828", ls="--", lw=1.2, label=f"FPE最小 p*={p_star}")
    ax.set_title("实验三(3)：真实EEG AR阶数-相对FPE曲线")
    ax.set_xlabel("AR阶数 p")
    ax.set_ylabel("相对FPE (最小值=1)")
    ax.legend()
    fpe_fig_path = save_figure(fig, "exp3_real_fpe.png")

    summary_path = TABLE_DIR / "exp3_real_fpe_summary.csv"
    pd.DataFrame([{"p_star": p_star, "min_fpe": float(fpe_df["fpe"].min()), "algorithm": "Yule-Walker"}]).to_csv(
        summary_path, index=False, encoding="utf-8-sig", float_format="%.15f"
    )
    return {
        "exp3_real_figure": spec_fig_path,
        "exp3_real_zoom_figure": zoom_fig_path,
        "exp3_real_table": metric_path,
        "exp3_real_risk_table": risk_path,
        "exp3_real_fpe_figure": fpe_fig_path,
        "exp3_real_fpe_summary": summary_path,
    }


if __name__ == "__main__":
    run()
