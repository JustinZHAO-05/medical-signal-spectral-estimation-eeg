from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "report"
GENERATED_DIR = REPORT_DIR / "generated"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
PDF_DIR = PROJECT_ROOT / "outputs" / "pdf"
FINAL_PDF = PDF_DIR / "实验二-谱估计-实验报告.pdf"


def main() -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    build_tables()
    compile_pdf()


def build_tables() -> None:
    write_table(
        "table_exp1_artificial.tex",
        "实验一人工信号四种谱估计的定量比较",
        "exp1_artificial_metrics.csv",
        [
            ("method", "方法"),
            ("peak_near_12_hz", "12 Hz峰(Hz)"),
            ("peak_12_db_hz", "12 Hz功率(dB/Hz)"),
            ("peak_near_24_hz", "24 Hz峰(Hz)"),
            ("peak_24_db_hz", "24 Hz功率(dB/Hz)"),
            ("harmonic_24_prominence_vs_15_30_db", "24 Hz突出度(dB)"),
            ("background_30_45_mean_db_hz", "30-45 Hz背景(dB/Hz)"),
            ("roughness_5_40_db", "曲线起伏"),
        ],
    )
    write_table(
        "table_exp1_real.tex",
        "实验一真实Oz 59 s EEG的alpha峰指标",
        "exp1_real_alpha_metrics.csv",
        [
            ("method", "方法"),
            ("alpha_peak_hz", "$\\alpha$峰(Hz)"),
            ("alpha_peak_db_hz", "$\\alpha$峰功率(dB/Hz)"),
            ("background_15_30_mean_db_hz", "15-30 Hz背景"),
            ("alpha_prominence_db", "突出度(dB)"),
        ],
    )
    write_table(
        "table_exp2_length.tex",
        "实验二数据长度与频率分辨率指标",
        "exp2_length_resolution_metrics.csv",
        [
            ("duration_s", "长度(s)"),
            ("theoretical_delta_f_hz", "1/T(Hz)"),
            ("target_10_peak_hz", "10 Hz峰"),
            ("target_11_peak_hz", "11 Hz峰"),
            ("valley_between_peaks_db_hz", "峰间最低(dB/Hz)"),
            ("weaker_peak_minus_valley_db", "弱峰-谷深(dB)"),
            ("main_peak_minus_3db_width_hz", "10 Hz峰宽(Hz)"),
            ("resolution_status", "分辨判定"),
        ],
    )
    write_table(
        "table_exp2_window.tex",
        "实验二窗函数泄漏指标",
        "exp2_window_leakage_metrics.csv",
        [
            ("window", "窗函数"),
            ("main_peak_hz", "强峰(Hz)"),
            ("main_minus_3db_width_hz", "强峰-3 dB宽度(Hz)"),
            ("weak_peak_hz", "弱峰(Hz)"),
            ("weak_peak_prominence_db", "弱峰突出度(dB)"),
        ],
    )
    write_table(
        "table_exp2_nfft.tex",
        "实验二NFFT频率轴间隔与分辨判断",
        "exp2_nfft_metrics.csv",
        [
            ("nfft", "nfft"),
            ("frequency_grid_interval_hz", "$\\Delta f=f_s/nfft$(Hz)"),
            ("rayleigh_interval_hz", "Rayleigh间隔(Hz)"),
            ("tone_spacing_hz", "真实频差(Hz)"),
            ("apparent_local_peak_count_9_12", "表观局部峰数"),
            ("physically_resolvable_two_peaks", "物理可分辨"),
        ],
        note="三种NFFT条件的数据长度均为1 s。即使补零后曲线上可出现局部极大值，10 Hz与10.4 Hz的真实频差仍小于1 Hz Rayleigh间隔，因此不能判定为稳定独立双峰。",
    )
    write_table(
        "table_exp2_real_length.tex",
        "实验二真实EEG不同时长alpha峰指标",
        "exp2_real_length_alpha_metrics.csv",
        [
            ("duration_s", "长度(s)"),
            ("nfft", "nfft"),
            ("frequency_grid_interval_hz", "$\\Delta f$(Hz)"),
            ("rayleigh_interval_hz", "1/T(Hz)"),
            ("alpha_peak_hz", "$\\alpha$峰(Hz)"),
            ("alpha_prominence_db", "突出度(dB)"),
            ("alpha_minus_3db_width_hz", "-3 dB宽度(Hz)"),
        ],
    )
    write_table(
        "table_exp2_real_window.tex",
        "实验二真实EEG窗函数alpha峰指标",
        "exp2_real_window_alpha_metrics.csv",
        [
            ("window", "窗函数"),
            ("alpha_peak_hz", "$\\alpha$峰(Hz)"),
            ("alpha_prominence_db", "突出度(dB)"),
            ("alpha_minus_3db_width_hz", "-3 dB宽度(Hz)"),
        ],
    )
    write_table(
        "table_exp2_real_nfft.tex",
        "实验二真实EEG同一30 s片段不同NFFT指标",
        "exp2_real_nfft_alpha_metrics.csv",
        [
            ("nfft", "nfft"),
            ("frequency_grid_interval_hz", "$\\Delta f$(Hz)"),
            ("alpha_peak_hz", "$\\alpha$峰(Hz)"),
            ("alpha_prominence_db", "突出度(dB)"),
            ("alpha_minus_3db_width_hz", "-3 dB宽度(Hz)"),
        ],
    )
    write_table(
        "table_exp3_artificial.tex",
        "实验三人工信号Welch与AR谱峰分辨指标",
        "exp3_artificial_welch_ar_metrics.csv",
        [
            ("method", "方法"),
            ("peak_count", "峰数"),
            ("peak_1_hz", "峰1(Hz)"),
            ("peak_2_hz", "峰2(Hz)"),
            ("valley_db_hz", "峰间谷(dB/Hz)"),
            ("resolvability_db", "可分辨性(dB)"),
        ],
    )
    write_table(
        "table_exp3_artificial_fpe.tex",
        "实验三人工信号FPE推荐阶数",
        "exp3_artificial_fpe_summary.csv",
        [("algorithm", "参数估计"), ("p_star", "$p^*$"), ("min_fpe", "最小FPE")],
        stretch_to_textwidth=False,
    )
    write_table(
        "table_exp3_artificial_risk.tex",
        "实验三人工信号AR阶数伪峰风险指标",
        "exp3_artificial_ar_peak_risk_metrics.csv",
        [
            ("method", "方法"),
            ("local_peak_count_6_18_hz", "6-18 Hz局部峰数"),
            ("local_peak_count_12_40_hz", "12-40 Hz局部峰数"),
            ("minus_3db_width_near_10hz", "10 Hz附近-3 dB宽度"),
        ],
    )
    write_table(
        "table_exp3_real.tex",
        "实验三真实Oz 8 s EEG Welch与AR谱alpha指标",
        "exp3_real_welch_ar_alpha_metrics.csv",
        [
            ("method", "方法"),
            ("alpha_peak_hz", "$\\alpha$峰(Hz)"),
            ("alpha_prominence_db", "突出度(dB)"),
            ("alpha_minus_3db_width_hz", "-3 dB宽度(Hz)"),
            ("p_star", "FPE p*"),
        ],
    )
    write_table(
        "table_exp3_real_risk.tex",
        "实验三真实EEG AR谱局部峰风险指标",
        "exp3_real_ar_peak_risk_metrics.csv",
        [
            ("method", "方法"),
            ("local_peak_count_alpha_8_13_hz", "$\\alpha$频段峰数"),
            ("local_peak_count_13_40_hz", "13-40 Hz局部峰数"),
            ("alpha_minus_3db_width_hz", "$\\alpha$峰-3 dB宽度"),
        ],
    )
    write_table(
        "table_exp3_real_fpe.tex",
        "实验三真实EEG FPE推荐阶数",
        "exp3_real_fpe_summary.csv",
        [("algorithm", "参数估计"), ("p_star", "$p^*$"), ("min_fpe", "最小FPE")],
        stretch_to_textwidth=False,
    )
    write_table(
        "table_exp4_realizations.tex",
        "实验四同一随机过程30次实现的跨实现稳定性",
        "exp4_artificial_realization_stability_metrics.csv",
        [
            ("method", "方法"),
            ("power_10hz_mean_db_hz", "10 Hz均值"),
            ("power_10hz_std_db", "10 Hz标准差"),
            ("alpha_peak_hz_mean", "$\\alpha$峰均值"),
            ("alpha_peak_hz_std", "$\\alpha$峰标准差"),
            ("alpha_prominence_mean_db", "突出度均值"),
            ("alpha_prominence_std_db", "突出度标准差"),
            ("pointwise_std_15_30_mean_db", "15-30 Hz点标准差"),
        ],
    )
    write_table(
        "table_exp4_segment.tex",
        "实验四Welch分段长度的稳定性与分辨率指标",
        "exp4_welch_segment_length_metrics.csv",
        [
            ("segment_length_s", "分段(s)"),
            ("power_10hz_std_db", "10 Hz标准差"),
            ("alpha_peak_hz_std", "$\\alpha$峰标准差"),
            ("alpha_prominence_mean_db", "突出度均值"),
            ("alpha_prominence_std_db", "突出度标准差"),
            ("pointwise_std_8_13_mean_db", "8-13 Hz点标准差"),
            ("mean_spectrum_alpha_minus_3db_width_hz", "均值谱-3 dB宽度"),
        ],
    )
    write_table(
        "table_exp4_real.tex",
        "实验四真实EEG 9个时间窗稳定性指标",
        "exp4_real_window_stability_metrics.csv",
        [
            ("method", "方法"),
            ("alpha_peak_hz_mean", "$\\alpha$峰均值"),
            ("alpha_peak_hz_std", "$\\alpha$峰标准差"),
            ("alpha_prominence_mean_db", "突出度均值"),
            ("alpha_prominence_std_db", "突出度标准差"),
            ("pointwise_std_8_13_mean_db", "8-13 Hz点标准差"),
            ("pointwise_std_15_30_mean_db", "15-30 Hz点标准差"),
            ("mean_spectrum_alpha_minus_3db_width_hz", "均值谱-3 dB宽度"),
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
        return "未分辨"
    if isinstance(value, str):
        return value
    try:
        v = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(v):
        return "未分辨"
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
            "report.tex",
        ],
        cwd=REPORT_DIR,
        check=True,
    )
    built_pdf = PDF_DIR / "report.pdf"
    if not built_pdf.exists():
        raise FileNotFoundError(built_pdf)
    shutil.copyfile(built_pdf, FINAL_PDF)
    print(f"Wrote {FINAL_PDF}")


if __name__ == "__main__":
    main()
