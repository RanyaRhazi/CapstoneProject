"""Audio helpers: decode uploaded files, resample to 16 kHz, and perturb audio for robustness tests."""
import io
from math import gcd

import numpy as np
from scipy.signal import resample_poly

TARGET_RATE = 16000


def to_mono_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Average channels to mono and resample to 16 kHz."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != TARGET_RATE:
        g = gcd(sample_rate, TARGET_RATE)
        audio = resample_poly(audio, TARGET_RATE // g, sample_rate // g).astype(np.float32)
    return audio


def decode(data: bytes) -> np.ndarray:
    """Decode WAV, FLAC or OGG bytes into mono 16 kHz float32 audio."""
    import soundfile as sf

    audio, sample_rate = sf.read(io.BytesIO(data), dtype="float32")
    return to_mono_16k(audio, sample_rate)


def add_noise(audio: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    """Add white Gaussian noise at the given signal-to-noise ratio (dB)."""
    audio = np.asarray(audio, dtype=np.float32)
    signal_power = float(np.mean(audio**2))
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.default_rng(seed).normal(0.0, np.sqrt(noise_power), audio.shape)
    return (audio + noise).astype(np.float32)


def change_speed(audio: np.ndarray, factor: float) -> np.ndarray:
    """Play audio `factor` times faster by resampling (pitch shifts too, as in Kaldi-style speed perturbation)."""
    audio = np.asarray(audio, dtype=np.float32)
    up, down = 100, int(round(100 * factor))
    g = gcd(up, down)
    return resample_poly(audio, up // g, down // g).astype(np.float32)
