from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .config import DATA_DIR, FS


@dataclass(frozen=True)
class EEGData:
    X: np.ndarray
    fs: float
    ch_names: list[str]
    source: str
    unit: str = "V"
    scale_factor: float = 1.0

    def channel(self, name: str) -> np.ndarray:
        return self.X[self.ch_names.index(name)]

    def segment(self, channel: str, start_s: float, end_s: float) -> np.ndarray:
        i0 = sec_to_sample(start_s, self.fs)
        i1 = sec_to_sample(end_s, self.fs)
        return self.channel(channel)[i0:i1].astype(np.float64, copy=True)


def sec_to_sample(t_sec: float, fs: float = FS) -> int:
    return int(round(float(t_sec) * float(fs)))


def load_eeg(data_dir: Path = DATA_DIR) -> EEGData:
    set_file = data_dir / "sub-032481_EC.set"

    if not set_file.is_file() or not set_file.with_suffix(".fdt").is_file():
        raise FileNotFoundError(f"Place the externally obtained .set/.fdt pair in {data_dir}, or set LEMON_EEG_DIR. See README.md.")

    try:
        import mne

        raw = mne.io.read_raw_eeglab(set_file, preload=True, verbose="ERROR")
        raw.pick("eeg", exclude=[])
        X = raw.get_data().astype(np.float64)
        fs = float(raw.info["sfreq"])
        ch_names = list(raw.ch_names)
        _validate_eeg(X, fs, ch_names)
        return EEGData(X=X, fs=fs, ch_names=ch_names, source="mne", unit="V", scale_factor=1.0)
    except Exception:
        return _load_eeglab_fdt_direct(set_file)


def _load_eeglab_fdt_direct(set_file: Path) -> EEGData:
    eeg = loadmat(set_file, squeeze_me=True, struct_as_record=False)["EEG"]
    n_channels = int(eeg.nbchan)
    n_times = int(eeg.pnts)
    fs = float(eeg.srate)
    ch_names = [str(chan.labels) for chan in np.atleast_1d(eeg.chanlocs)]

    fdt_file = set_file.with_suffix(".fdt")
    raw = np.fromfile(fdt_file, dtype="<f4")
    expected = n_channels * n_times
    if raw.size != expected:
        raise ValueError(f"Unexpected .fdt size: got {raw.size}, expected {expected}.")

    # EEGLAB stores these preprocessed amplitudes in microvolts. MNE returns
    # data in volts, so the direct fallback uses the same unit to keep absolute
    # PSD levels comparable across loaders.
    scale_factor = 1e-6
    X = raw.reshape((n_channels, n_times), order="F").astype(np.float64) * scale_factor
    _validate_eeg(X, fs, ch_names)
    return EEGData(X=X, fs=fs, ch_names=ch_names, source="scipy-loadmat+fdt", unit="V", scale_factor=scale_factor)


def _validate_eeg(X: np.ndarray, fs: float, ch_names: list[str]) -> None:
    if X.ndim != 2:
        raise ValueError(f"EEG array must be 2D, got shape {X.shape}.")
    if abs(fs - FS) > 1e-9:
        raise ValueError(f"Expected fs={FS}, got {fs}.")
    if len(ch_names) != X.shape[0]:
        raise ValueError("Channel name count does not match EEG data rows.")
    if "Oz" not in ch_names:
        raise ValueError("Oz channel was not found in the EEG data.")
