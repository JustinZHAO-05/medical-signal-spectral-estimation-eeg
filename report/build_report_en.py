from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "report"
GENERATED_DIR = REPORT_DIR / "generated_en"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
PDF_DIR = PROJECT_ROOT / "outputs" / "pdf"
FINAL_PDF = PDF_DIR / "Spectral_Estimation_Lab_Report_EN.pdf"

VALUE_TRANSLATIONS = {
    "直接法(矩形窗FFT)": "Direct FFT, rectangular window",
    "周期图法(Hamming窗)": "Periodogram, Hamming window",
    "自相关法": "Autocorrelation method",
    "Welch法(1s/50%)": "Welch, 1 s / 50%",
    "单次周期图": "Single periodogram",
    "矩形窗": "Rectangular",
    "Hann窗": "Hann",
    "Hamming窗": "Hamming",
    "Blackman窗": "Blackman",
    "临界可分": "marginally resolved",
    "清晰可分": "clearly resolved",
    "否": "No",
    "是": "Yes",
    "未分辨": "not resolved",
}


def main() -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    build_tables()
    compile_pdf()


def build_tables() -> None:
    write_table(
        "table_exp1_artificial.tex",
        "Quantitative comparison of four PSD estimates for the artificial signal",
        "exp1_artificial_metrics.csv",
        [
            ("method", "Method"),
            ("peak_near_12_hz", "12 Hz peak"),
            ("peak_12_db_hz", "12 Hz power"),
            ("peak_near_24_hz", "24 Hz peak"),
            ("peak_24_db_hz", "24 Hz power"),
            ("harmonic_24_prominence_vs_15_30_db", "24 Hz prominence"),
            ("roughness_5_40_db", "roughness"),
        ],
    )
    write_table(
        "table_exp1_real.tex",
        "Alpha-peak metrics for the 59 s real Oz EEG segment",
        "exp1_real_alpha_metrics.csv",
        [
            ("method", "Method"),
            ("alpha_peak_hz", r"$\alpha$ peak (Hz)"),
            ("alpha_peak_db_hz", r"$\alpha$ power (dB/Hz)"),
            ("background_15_30_mean_db_hz", "15-30 Hz background"),
            ("alpha_prominence_db", "prominence (dB)"),
        ],
    )
    write_table(
        "table_exp2_length.tex",
        "Record length and frequency-resolution metrics",
        "exp2_length_resolution_metrics.csv",
        [
            ("duration_s", "Length (s)"),
            ("theoretical_delta_f_hz", "1/T (Hz)"),
            ("target_10_peak_hz", "10 Hz peak"),
            ("target_11_peak_hz", "11 Hz peak"),
            ("valley_between_peaks_db_hz", "valley (dB/Hz)"),
            ("weaker_peak_minus_valley_db", "weak peak - valley"),
            ("main_peak_minus_3db_width_hz", "10 Hz width"),
            ("resolution_status", "resolution"),
        ],
    )
    write_table(
        "table_exp2_window.tex",
        "Window leakage metrics",
        "exp2_window_leakage_metrics.csv",
        [
            ("window", "Window"),
            ("main_peak_hz", "strong peak"),
            ("main_minus_3db_width_hz", "-3 dB width"),
            ("weak_peak_hz", "weak peak"),
            ("weak_peak_prominence_db", "weak-peak prominence"),
        ],
    )
    write_table(
        "table_exp2_nfft.tex",
        "NFFT grid spacing and resolution judgment",
        "exp2_nfft_metrics.csv",
        [
            ("nfft", "nfft"),
            ("frequency_grid_interval_hz", r"$\Delta f=f_s/nfft$"),
            ("rayleigh_interval_hz", "Rayleigh interval"),
            ("tone_spacing_hz", "tone spacing"),
            ("apparent_local_peak_count_9_12", "apparent peaks"),
            ("physically_resolvable_two_peaks", "physically resolved"),
        ],
        note="All three NFFT conditions use the same 1 s record. Zero padding may create more sampled points and local maxima, but 10 Hz and 10.4 Hz are still closer than the 1 Hz Rayleigh interval.",
    )
    write_table(
        "table_exp2_real_length.tex",
        "Real EEG alpha metrics under different record lengths",
        "exp2_real_length_alpha_metrics.csv",
        [
            ("duration_s", "Length (s)"),
            ("nfft", "nfft"),
            ("frequency_grid_interval_hz", r"$\Delta f$"),
            ("rayleigh_interval_hz", "1/T"),
            ("alpha_peak_hz", r"$\alpha$ peak"),
            ("alpha_prominence_db", "prominence"),
            ("alpha_minus_3db_width_hz", "-3 dB width"),
        ],
    )
    write_table(
        "table_exp2_real_window.tex",
        "Real EEG alpha metrics under different windows",
        "exp2_real_window_alpha_metrics.csv",
        [
            ("window", "Window"),
            ("alpha_peak_hz", r"$\alpha$ peak"),
            ("alpha_prominence_db", "prominence"),
            ("alpha_minus_3db_width_hz", "-3 dB width"),
        ],
    )
    write_table(
        "table_exp2_real_nfft.tex",
        "Real EEG alpha metrics for the same 30 s segment under different NFFT values",
        "exp2_real_nfft_alpha_metrics.csv",
        [
            ("nfft", "nfft"),
            ("frequency_grid_interval_hz", r"$\Delta f$"),
            ("alpha_peak_hz", r"$\alpha$ peak"),
            ("alpha_prominence_db", "prominence"),
            ("alpha_minus_3db_width_hz", "-3 dB width"),
        ],
    )
    write_table(
        "table_exp3_artificial.tex",
        "Welch and AR peak-resolution metrics for the artificial signal",
        "exp3_artificial_welch_ar_metrics.csv",
        [
            ("method", "Method"),
            ("peak_count", "peaks"),
            ("peak_1_hz", "peak 1"),
            ("peak_2_hz", "peak 2"),
            ("valley_db_hz", "valley"),
            ("resolvability_db", "resolvability"),
        ],
    )
    write_table(
        "table_exp3_artificial_fpe.tex",
        "FPE-selected AR order for the artificial signal",
        "exp3_artificial_fpe_summary.csv",
        [("algorithm", "Estimator"), ("p_star", r"$p^*$"), ("min_fpe", "minimum FPE")],
        stretch_to_textwidth=False,
    )
    write_table(
        "table_exp3_artificial_risk.tex",
        "AR-order pseudo-peak risk metrics for the artificial signal",
        "exp3_artificial_ar_peak_risk_metrics.csv",
        [
            ("method", "Method"),
            ("local_peak_count_6_18_hz", "6-18 Hz peaks"),
            ("local_peak_count_12_40_hz", "12-40 Hz peaks"),
            ("minus_3db_width_near_10hz", "10 Hz -3 dB width"),
        ],
    )
    write_table(
        "table_exp3_real.tex",
        "Welch and AR alpha metrics for the real Oz EEG segment",
        "exp3_real_welch_ar_alpha_metrics.csv",
        [
            ("method", "Method"),
            ("alpha_peak_hz", r"$\alpha$ peak"),
            ("alpha_prominence_db", "prominence"),
            ("alpha_minus_3db_width_hz", "-3 dB width"),
            ("p_star", r"FPE $p^*$"),
        ],
    )
    write_table(
        "table_exp3_real_risk.tex",
        "AR local-peak risk metrics for the real EEG segment",
        "exp3_real_ar_peak_risk_metrics.csv",
        [
            ("method", "Method"),
            ("local_peak_count_alpha_8_13_hz", r"$\alpha$-band peaks"),
            ("local_peak_count_13_40_hz", "13-40 Hz peaks"),
            ("alpha_minus_3db_width_hz", r"$\alpha$ -3 dB width"),
        ],
    )
    write_table(
        "table_exp3_real_fpe.tex",
        "FPE-selected AR order for the real EEG segment",
        "exp3_real_fpe_summary.csv",
        [("algorithm", "Estimator"), ("p_star", r"$p^*$"), ("min_fpe", "minimum FPE")],
        stretch_to_textwidth=False,
    )
    write_table(
        "table_exp4_realizations.tex",
        "Cross-realization stability for 30 realizations of the same random process",
        "exp4_artificial_realization_stability_metrics.csv",
        [
            ("method", "Method"),
            ("power_10hz_mean_db_hz", "10 Hz mean"),
            ("power_10hz_std_db", "10 Hz std"),
            ("alpha_peak_hz_mean", r"$\alpha$ peak mean"),
            ("alpha_peak_hz_std", r"$\alpha$ peak std"),
            ("alpha_prominence_mean_db", "prom. mean"),
            ("alpha_prominence_std_db", "prom. std"),
            ("pointwise_std_15_30_mean_db", "15-30 Hz pointwise std"),
        ],
    )
    write_table(
        "table_exp4_segment.tex",
        "Welch segment-length stability and resolution metrics",
        "exp4_welch_segment_length_metrics.csv",
        [
            ("segment_length_s", "segment (s)"),
            ("power_10hz_std_db", "10 Hz std"),
            ("alpha_peak_hz_std", r"$\alpha$ peak std"),
            ("alpha_prominence_mean_db", "prom. mean"),
            ("alpha_prominence_std_db", "prom. std"),
            ("pointwise_std_8_13_mean_db", "8-13 Hz pointwise std"),
            ("mean_spectrum_alpha_minus_3db_width_hz", "mean-spectrum width"),
        ],
    )
    write_table(
        "table_exp4_real.tex",
        "Stability metrics across nine real EEG windows",
        "exp4_real_window_stability_metrics.csv",
        [
            ("method", "Method"),
            ("alpha_peak_hz_mean", r"$\alpha$ peak mean"),
            ("alpha_peak_hz_std", r"$\alpha$ peak std"),
            ("alpha_prominence_mean_db", "prom. mean"),
            ("alpha_prominence_std_db", "prom. std"),
            ("pointwise_std_8_13_mean_db", "8-13 Hz pointwise std"),
            ("pointwise_std_15_30_mean_db", "15-30 Hz pointwise std"),
            ("mean_spectrum_alpha_minus_3db_width_hz", "mean-spectrum width"),
        ],
    )


def write_table(
    filename: str,
    caption: str,
    csv_name: str,
    columns: list[tuple[str, str]],
    note: str | None = None,
    stretch_to_textwidth: bool = True,
) -> None:
    df = pd.read_csv(TABLE_DIR / csv_name)
    selected = df[[col for col, _ in columns]].copy()
    headers = [title for _, title in columns]
    body_rows = []
    for _, row in selected.iterrows():
        body_rows.append(" & ".join(latex_escape(format_value(row[col])) for col, _ in columns) + r" \\")

    align = "l" * len(columns)
    tabular_lines = [
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(latex_header(h) for h in headers) + r" \\",
        r"\midrule",
        *body_rows,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    if stretch_to_textwidth:
        table_body = [r"\resizebox{\textwidth}{!}{%", *tabular_lines, r"}"]
    else:
        table_body = tabular_lines

    text = "\n".join(
        [
            r"\begin{table}[htbp]",
            r"\centering",
            rf"\caption{{{latex_escape(caption)}}}",
            r"\small",
            *table_body,
            rf"\par\footnotesize {latex_escape(note)}" if note else "",
            r"\end{table}",
            "",
        ]
    )
    (GENERATED_DIR / filename).write_text(text, encoding="utf-8")


def format_value(value) -> str:
    if pd.isna(value):
        return "not resolved"
    if isinstance(value, str):
        return VALUE_TRANSLATIONS.get(value, value)
    try:
        v = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(v):
        return "not resolved"
    if v != 0 and abs(v) < 0.001:
        return "<0.001" if v > 0 else ">-0.001"
    if abs(v - round(v)) < 1e-10 and abs(v) < 100000:
        return str(int(round(v)))
    if abs(v) >= 10000:
        return f"{v:.3f}".rstrip("0").rstrip(".")
    return f"{v:.3f}"


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(value))


def latex_header(value: str) -> str:
    value = str(value)
    if "$" in value or "\\" in value:
        return value
    return latex_escape(value)


def compile_pdf() -> None:
    subprocess.run(
        [
            "latexmk",
            "-g",
            "-xelatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-outdir=../outputs/pdf",
            "report_en.tex",
        ],
        cwd=REPORT_DIR,
        check=True,
    )
    built_pdf = PDF_DIR / "report_en.pdf"
    if not built_pdf.exists():
        raise FileNotFoundError(built_pdf)
    shutil.copyfile(built_pdf, FINAL_PDF)
    print(f"Wrote {FINAL_PDF}")


if __name__ == "__main__":
    main()
