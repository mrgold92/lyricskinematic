"""Análisis de audio con numpy puro: energía por bandas, espectro, beats y tonalidad.

No usa librosa/numba a propósito: sus DLL pueden quedar bloqueadas por el Control de
aplicaciones de Windows y numpy + soundfile bastan para lo que necesitamos."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import soundfile as sf

N_BARS = 64
_HOP = 512
_N_FFT = 2048
_SR = 22050


@dataclass
class Features:
    fps: int
    n_frames: int
    duration: float
    tempo: float
    key_hue: float
    rms: np.ndarray        # (n_frames,) 0..1
    bass: np.ndarray
    mid: np.ndarray
    high: np.ndarray
    beat_env: np.ndarray   # 1 en cada beat, decae rápido
    beat_times: np.ndarray
    bars: np.ndarray       # (n_frames, N_BARS) 0..1
    phase: np.ndarray      # tiempo "elástico": avanza más rápido con la energía


# ------------------------------------------------------------------ utilidades

def _load_mono(path: str) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr == 2 * _SR:                       # 44100 -> 22050: media de pares (paso bajo barato)
        n = len(y) // 2 * 2
        return (y[:n:2] + y[1:n:2]) / 2
    if sr != _SR:
        x = np.arange(int(len(y) * _SR / sr)) * (sr / _SR)
        return np.interp(x, np.arange(len(y)), y).astype(np.float32)
    return y


def _mel(f: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_filters(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Banco de filtros triangulares (n_mels, 1 + n_fft // 2)."""
    freqs = np.linspace(0, sr / 2, 1 + n_fft // 2)
    pts = 700.0 * (10 ** (np.linspace(_mel(np.array(fmin)), _mel(np.array(fmax)), n_mels + 2) / 2595.0) - 1.0)
    fb = np.zeros((n_mels, len(freqs)), dtype=np.float32)
    for i in range(n_mels):
        lo, mid, hi = pts[i], pts[i + 1], pts[i + 2]
        up = (freqs - lo) / max(mid - lo, 1e-6)
        down = (hi - freqs) / max(hi - mid, 1e-6)
        fb[i] = np.maximum(0, np.minimum(up, down))
    fb /= np.maximum(fb.sum(axis=1, keepdims=True), 1e-9)       # energía comparable entre bandas
    return fb


def _stft_features(y: np.ndarray, mel_fb: np.ndarray, chroma_map: np.ndarray):
    """Devuelve (mel_power (frames, n_mels), rms (frames,), chroma (12,)), procesando por bloques."""
    pad = _N_FFT // 2
    yp = np.pad(y, (pad, pad), mode="reflect")
    n_frames = 1 + (len(yp) - _N_FFT) // _HOP
    win = np.hanning(_N_FFT).astype(np.float32)
    mel = np.empty((n_frames, mel_fb.shape[0]), dtype=np.float32)
    rms = np.empty(n_frames, dtype=np.float32)
    chroma = np.zeros(12, dtype=np.float64)
    view = np.lib.stride_tricks.sliding_window_view(yp, _N_FFT)[::_HOP]
    for a in range(0, n_frames, 1500):
        frames = view[a:a + 1500]
        rms[a:a + 1500] = np.sqrt((frames ** 2).mean(axis=1))
        power = np.abs(np.fft.rfft(frames * win, axis=1)) ** 2
        mel[a:a + 1500] = power @ mel_fb.T
        chroma += (power @ chroma_map).sum(axis=0)
    return mel, rms, chroma


def _smooth(x: np.ndarray, attack: float = 0.55, release: float = 0.12) -> np.ndarray:
    out = np.empty_like(x)
    v = np.zeros(x.shape[1:], dtype=x.dtype)
    for i in range(len(x)):
        v = v + (x[i] - v) * np.where(x[i] > v, attack, release)
        out[i] = v
    return out


def _norm(x: np.ndarray, pct: float = 98.0) -> np.ndarray:
    return np.clip(x / max(float(np.percentile(x, pct)), 1e-6), 0, 1)


def _norm_cols(bars: np.ndarray) -> np.ndarray:
    """Normaliza cada banda por separado para que los agudos también se vean."""
    top = np.percentile(bars, 97, axis=0)
    return np.clip(bars / np.maximum(top, 1e-3), 0, 1)


def _beat_track(onset: np.ndarray, fps_onset: float) -> tuple[float, np.ndarray]:
    """Tempo por autocorrelación + seguimiento de beats por programación dinámica (Ellis)."""
    env = onset - onset.mean()
    env = env / max(env.std(), 1e-6)
    n = len(env)
    ac = np.correlate(env, env, mode="full")[n - 1:]
    lags = np.arange(len(ac))
    bpm = 60.0 * fps_onset / np.maximum(lags, 1)
    prior = np.exp(-0.5 * (np.log2(np.maximum(bpm, 1e-6) / 120.0) / 1.0) ** 2)
    score = ac * prior
    score[(bpm < 60) | (bpm > 200)] = 0
    period = float(np.argmax(score))
    if period < 2:
        return 0.0, np.array([])
    tempo = 60.0 * fps_onset / period

    # programación dinámica: cada beat debe caer cerca de un período del anterior
    alpha = 100.0
    cum = env.copy()
    back = np.full(n, -1, dtype=int)
    lo, hi = int(round(period / 2)), int(round(period * 2))
    prange = np.arange(lo, hi + 1)
    penalty = -alpha * (np.log(prange / period)) ** 2
    for t in range(n):
        prev = t - prange
        ok = prev >= 0
        if ok.any():
            cand = cum[prev[ok]] + penalty[ok]
            j = int(np.argmax(cand))
            if cand[j] > 0:
                cum[t] = env[t] + cand[j]
                back[t] = prev[ok][j]
    # se parte del mejor máximo local en el último tramo
    tail = cum[max(0, n - int(period * 2)):]
    last = int(np.argmax(tail)) + max(0, n - int(period * 2))
    beats = [last]
    while back[beats[-1]] >= 0:
        beats.append(int(back[beats[-1]]))
    return tempo, np.array(beats[::-1]) * _HOP / _SR


# ------------------------------------------------------------------ análisis

def analyze(wav_path: str, fps: int) -> Features:
    y = _load_mono(wav_path)
    duration = len(y) / _SR
    n_frames = int(math.ceil(duration * fps))

    mel_fb = _mel_filters(_SR, _N_FFT, N_BARS, 0.0, 10000.0)
    # mapa bin FFT -> clase de altura (65 Hz..2 kHz)
    freqs = np.linspace(0, _SR / 2, 1 + _N_FFT // 2)
    chroma_map = np.zeros((len(freqs), 12), dtype=np.float32)
    sel = (freqs >= 65) & (freqs <= 2000)
    pc = np.round(12 * np.log2(freqs[sel] / 440.0) + 69).astype(int) % 12
    chroma_map[np.nonzero(sel)[0], pc] = 1.0

    mel, rms_raw, chroma = _stft_features(y, mel_fb, chroma_map)
    db = 10 * np.log10(np.maximum(mel, 1e-10))
    db -= db.max()
    level = np.clip((db + 60.0) / 60.0, 0, 1)                      # (frames, bins)
    idx = np.clip((np.arange(n_frames) / fps * _SR / _HOP).astype(int), 0, level.shape[0] - 1)
    bars = _smooth(_norm_cols(level[idx].astype(np.float32)))

    bass = _smooth(_norm(bars[:, :6].mean(axis=1))[:, None])[:, 0]
    mid = _smooth(_norm(bars[:, 6:30].mean(axis=1))[:, None])[:, 0]
    high = _smooth(_norm(bars[:, 30:].mean(axis=1))[:, None])[:, 0]
    rms = _smooth(_norm(rms_raw[idx])[:, None])[:, 0]

    # ritmo: flujo espectral (aumentos de energía por banda) -> tempo y beats
    onset = np.maximum(0, np.diff(db, axis=0, prepend=db[:1])).mean(axis=1)
    tempo, beat_times = _beat_track(onset, _SR / _HOP)
    times = np.arange(n_frames) / fps
    if len(beat_times):
        last = np.searchsorted(beat_times, times, side="right") - 1
        since = np.where(last >= 0, times - beat_times[np.clip(last, 0, None)], 9.0)
    else:
        since = np.full(n_frames, 9.0)
    beat_env = np.exp(-since / 0.16).astype(np.float32)

    key_hue = float(np.argmax(chroma)) / 12.0
    speed = 0.5 + 1.4 * rms + 0.8 * beat_env
    phase = np.cumsum(speed) / fps

    return Features(
        fps=fps, n_frames=n_frames, duration=duration, tempo=float(tempo), key_hue=key_hue,
        rms=rms.astype(np.float32), bass=bass.astype(np.float32),
        mid=mid.astype(np.float32), high=high.astype(np.float32),
        beat_env=beat_env, beat_times=np.asarray(beat_times), bars=bars.astype(np.float32),
        phase=phase.astype(np.float32),
    )
