from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import signal
from scipy.linalg import solve_toeplitz


EPS = np.finfo(float).tiny


@dataclass(frozen=True)
class PSD:
    freqs: np.ndarray
    pxx: np.ndarray
    label: str

    @property
    def db(self) -> np.ndarray:
        return to_db(self.pxx)


def to_db(pxx: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(pxx, dtype=float), EPS))


def demean(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x - np.mean(x)


def _window_values(window: str, n: int) -> np.ndarray:
    if window in {"boxcar", "rect", "rectangular", "none"}:
        return np.ones(n, dtype=float)
    return signal.get_window(window, n, fftbins=True).astype(float)


def one_sided_from_fft(y: np.ndarray, fs: float, nfft: int, scale_sum_sq: float) -> tuple[np.ndarray, np.ndarray]:
    spec = np.fft.rfft(y, n=nfft)
    pxx = (np.abs(spec) ** 2) / (fs * scale_sum_sq)
    if nfft % 2 == 0:
        pxx[1:-1] *= 2.0
    else:
        pxx[1:] *= 2.0
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    return freqs, pxx


def direct_psd(x: np.ndarray, fs: float, nfft: int | None = None, label: str = "直接法") -> PSD:
    x = demean(x)
    nfft = int(nfft or len(x))
    w = np.ones(len(x), dtype=float)
    freqs, pxx = one_sided_from_fft(x * w, fs, nfft, np.sum(w**2))
    return PSD(freqs=freqs, pxx=pxx, label=label)


def periodogram_psd(
    x: np.ndarray,
    fs: float,
    window: str = "hamming",
    nfft: int | None = None,
    label: str | None = None,
) -> PSD:
    x = demean(x)
    nfft = int(nfft or len(x))
    w = _window_values(window, len(x))
    freqs, pxx = one_sided_from_fft(x * w, fs, nfft, np.sum(w**2))
    return PSD(freqs=freqs, pxx=pxx, label=label or f"周期图-{window}")


def autocorrelation_psd(x: np.ndarray, fs: float, nfft: int | None = None, label: str = "自相关法") -> PSD:
    x = demean(x)
    n = len(x)
    min_nfft = 2 * n
    nfft = int(max(nfft or min_nfft, min_nfft))
    acf = signal.correlate(x, x, mode="full", method="fft") / n
    mid = n - 1
    seq = np.zeros(nfft, dtype=float)
    seq[0:n] = acf[mid:]
    seq[-(n - 1) :] = acf[:mid]
    pxx = np.real(np.fft.rfft(seq)) / fs
    pxx = np.maximum(pxx, EPS)
    if nfft % 2 == 0:
        pxx[1:-1] *= 2.0
    else:
        pxx[1:] *= 2.0
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    return PSD(freqs=freqs, pxx=pxx, label=label)


def welch_psd(
    x: np.ndarray,
    fs: float,
    segment_s: float,
    overlap: float = 0.5,
    window: str = "hamming",
    nfft: int | None = None,
    label: str = "Welch",
) -> PSD:
    nperseg = int(round(segment_s * fs))
    noverlap = int(round(nperseg * overlap))
    nfft = int(nfft or nperseg)
    freqs, pxx = signal.welch(
        demean(x),
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=False,
        return_onesided=True,
        scaling="density",
    )
    return PSD(freqs=freqs, pxx=pxx, label=label)


def yule_walker(x: np.ndarray, order: int) -> tuple[np.ndarray, float]:
    x = demean(x)
    n = len(x)
    if order >= n - 1:
        raise ValueError("AR order must be less than len(x)-1.")
    r_full = signal.correlate(x, x, mode="full", method="fft") / n
    r = r_full[n - 1 : n + order]
    coeffs = solve_toeplitz((r[:order], r[:order]), -r[1 : order + 1])
    noise_var = float(r[0] + np.dot(coeffs, r[1 : order + 1]))
    return coeffs.astype(float), max(noise_var, EPS)


def ar_psd(
    x: np.ndarray,
    fs: float,
    order: int,
    nfft: int = 4096,
    label: str | None = None,
) -> PSD:
    coeffs, noise_var = yule_walker(x, order)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    omega = 2.0 * np.pi * freqs / fs
    k = np.arange(1, order + 1, dtype=float)
    denom = np.abs(1.0 + np.exp(-1j * np.outer(omega, k)) @ coeffs) ** 2
    pxx = noise_var / (fs * np.maximum(denom, EPS))
    if nfft % 2 == 0:
        pxx[1:-1] *= 2.0
    else:
        pxx[1:] *= 2.0
    return PSD(freqs=freqs, pxx=pxx, label=label or f"AR(p={order})")


def fpe_values(x: np.ndarray, orders: Iterable[int]) -> list[dict[str, float]]:
    n = len(x)
    rows: list[dict[str, float]] = []
    for order in orders:
        _, noise_var = yule_walker(x, int(order))
        if n - order - 1 <= 0:
            fpe = np.nan
        else:
            fpe = noise_var * (n + order + 1) / (n - order - 1)
        rows.append({"order": int(order), "noise_variance": noise_var, "fpe": float(fpe)})
    return rows


def band_mask(freqs: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    return (freqs >= band[0]) & (freqs <= band[1])


def band_peak(freqs: np.ndarray, pxx_db: np.ndarray, band: tuple[float, float]) -> tuple[float, float]:
    mask = band_mask(freqs, band)
    if not np.any(mask):
        return np.nan, np.nan
    idxs = np.flatnonzero(mask)
    idx = idxs[np.argmax(pxx_db[mask])]
    return float(freqs[idx]), float(pxx_db[idx])


def band_mean_db(freqs: np.ndarray, pxx_db: np.ndarray, band: tuple[float, float]) -> float:
    mask = band_mask(freqs, band)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(pxx_db[mask]))


def band_std_db(freqs: np.ndarray, pxx_db: np.ndarray, band: tuple[float, float]) -> float:
    mask = band_mask(freqs, band)
    if not np.any(mask):
        return float("nan")
    return float(np.std(pxx_db[mask], ddof=1))


def pointwise_std_db(psds: list[PSD], band: tuple[float, float]) -> float:
    """Mean pointwise standard deviation across spectra in a frequency band."""
    if not psds:
        return float("nan")
    freqs = psds[0].freqs
    mask = band_mask(freqs, band)
    if not np.any(mask):
        return float("nan")
    stack = np.vstack([psd.db for psd in psds])
    return float(np.mean(np.std(stack[:, mask], axis=0, ddof=1)))


def rayleigh_interval(duration_s: float) -> float:
    return 1.0 / float(duration_s)


def relative_valley_depth(
    freqs: np.ndarray,
    pxx_db: np.ndarray,
    left_peak_band: tuple[float, float],
    right_peak_band: tuple[float, float],
) -> dict[str, float]:
    left_f, left_db = band_peak(freqs, pxx_db, left_peak_band)
    right_f, right_db = band_peak(freqs, pxx_db, right_peak_band)
    if not np.isfinite(left_f) or not np.isfinite(right_f):
        return {
            "left_peak_hz": left_f,
            "right_peak_hz": right_f,
            "left_peak_db_hz": left_db,
            "right_peak_db_hz": right_db,
            "valley_between_peaks_db_hz": float("nan"),
            "weaker_peak_minus_valley_db": float("nan"),
        }
    lo, hi = sorted((left_f, right_f))
    mask = (freqs >= lo) & (freqs <= hi)
    valley = float(np.min(pxx_db[mask])) if np.any(mask) else float("nan")
    weaker = min(left_db, right_db)
    return {
        "left_peak_hz": left_f,
        "right_peak_hz": right_f,
        "left_peak_db_hz": left_db,
        "right_peak_db_hz": right_db,
        "valley_between_peaks_db_hz": valley,
        "weaker_peak_minus_valley_db": weaker - valley,
    }


def count_local_maxima(
    freqs: np.ndarray,
    pxx_db: np.ndarray,
    band: tuple[float, float],
    prominence_db: float = 1.0,
    min_distance_hz: float = 0.2,
) -> int:
    mask = band_mask(freqs, band)
    f_band = freqs[mask]
    y_band = pxx_db[mask]
    if len(f_band) < 3:
        return 0
    df = float(np.median(np.diff(f_band)))
    distance = max(1, int(round(min_distance_hz / df)))
    peaks, _ = signal.find_peaks(y_band, prominence=prominence_db, distance=distance)
    return int(len(peaks))


def alpha_metrics(freqs: np.ndarray, pxx_db: np.ndarray) -> dict[str, float]:
    peak_f, peak_db = band_peak(freqs, pxx_db, (8.0, 13.0))
    background_db = band_mean_db(freqs, pxx_db, (15.0, 30.0))
    width = minus_3db_width(freqs, pxx_db, (8.0, 13.0))
    return {
        "alpha_peak_hz": peak_f,
        "alpha_peak_db_hz": peak_db,
        "background_15_30_mean_db_hz": background_db,
        "alpha_prominence_db": peak_db - background_db,
        "alpha_minus_3db_width_hz": width,
    }


def minus_3db_width(
    freqs: np.ndarray,
    pxx_db: np.ndarray,
    peak_band: tuple[float, float],
    search_band: tuple[float, float] | None = None,
) -> float:
    peak_f, peak_db = band_peak(freqs, pxx_db, peak_band)
    if not np.isfinite(peak_f):
        return float("nan")
    target = peak_db - 3.0
    peak_idx = int(np.argmin(np.abs(freqs - peak_f)))
    if search_band is None:
        search_band = (freqs[0], freqs[-1])
    valid = band_mask(freqs, search_band)

    left_idx = None
    for i in range(peak_idx - 1, 0, -1):
        if not valid[i]:
            break
        if pxx_db[i] <= target <= pxx_db[i + 1]:
            left_idx = i
            break

    right_idx = None
    for i in range(peak_idx + 1, len(freqs) - 1):
        if not valid[i]:
            break
        if pxx_db[i] <= target <= pxx_db[i - 1]:
            right_idx = i
            break

    if left_idx is None or right_idx is None:
        return float("nan")

    left_f = _interp_crossing(freqs[left_idx], pxx_db[left_idx], freqs[left_idx + 1], pxx_db[left_idx + 1], target)
    right_f = _interp_crossing(freqs[right_idx - 1], pxx_db[right_idx - 1], freqs[right_idx], pxx_db[right_idx], target)
    return float(right_f - left_f)


def _interp_crossing(f0: float, y0: float, f1: float, y1: float, target: float) -> float:
    if abs(y1 - y0) < 1e-12:
        return float((f0 + f1) / 2.0)
    return float(f0 + (target - y0) * (f1 - f0) / (y1 - y0))


def peak_resolution_metrics(
    freqs: np.ndarray,
    pxx_db: np.ndarray,
    band: tuple[float, float] = (9.0, 12.0),
    prominence_db: float = 3.0,
    min_distance_hz: float = 0.25,
) -> dict[str, float | str]:
    mask = band_mask(freqs, band)
    f_band = freqs[mask]
    y_band = pxx_db[mask]
    if len(f_band) < 3:
        return _empty_peak_resolution()
    df = float(np.median(np.diff(f_band)))
    distance = max(1, int(round(min_distance_hz / df)))
    peaks, props = signal.find_peaks(y_band, prominence=prominence_db, distance=distance)
    peak_count = int(len(peaks))
    if peak_count == 0:
        return _empty_peak_resolution(peak_count=0)

    order = np.argsort(y_band[peaks])[::-1]
    selected = peaks[order[:2]]
    selected = selected[np.argsort(f_band[selected])]
    peak_freqs = [float(f_band[i]) for i in selected]
    peak_vals = [float(y_band[i]) for i in selected]

    valley_db = float("nan")
    resolvability = float("nan")
    if len(selected) == 2:
        lo, hi = selected[0], selected[1]
        valley_db = float(np.min(y_band[lo : hi + 1]))
        resolvability = float(min(peak_vals) - valley_db)

    return {
        "peak_count": peak_count,
        "peak_1_hz": peak_freqs[0] if len(peak_freqs) > 0 else np.nan,
        "peak_2_hz": peak_freqs[1] if len(peak_freqs) > 1 else np.nan,
        "peak_1_db_hz": peak_vals[0] if len(peak_vals) > 0 else np.nan,
        "peak_2_db_hz": peak_vals[1] if len(peak_vals) > 1 else np.nan,
        "valley_db_hz": valley_db,
        "resolvability_db": resolvability,
    }


def _empty_peak_resolution(peak_count: int = 0) -> dict[str, float]:
    return {
        "peak_count": peak_count,
        "peak_1_hz": np.nan,
        "peak_2_hz": np.nan,
        "peak_1_db_hz": np.nan,
        "peak_2_db_hz": np.nan,
        "valley_db_hz": np.nan,
        "resolvability_db": np.nan,
    }


def value_at(freqs: np.ndarray, y: np.ndarray, f0: float) -> float:
    return float(np.interp(f0, freqs, y))


def roughness_db(freqs: np.ndarray, pxx_db: np.ndarray, band: tuple[float, float] = (5.0, 40.0)) -> float:
    mask = band_mask(freqs, band)
    y = pxx_db[mask]
    if len(y) < 2:
        return float("nan")
    return float(np.std(np.diff(y)))
