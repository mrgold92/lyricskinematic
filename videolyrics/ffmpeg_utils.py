"""Utilidades de ffmpeg (usa el binario incluido en imageio-ffmpeg)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import imageio_ffmpeg


def ffmpeg_exe() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg falló:\n{' '.join(cmd)}\n{proc.stderr[-2000:]}")


def decode_to_wav(src: Path, dst: Path, sr: int = 44100, channels: int = 2) -> None:
    """Convierte cualquier formato de audio a WAV."""
    _run([ffmpeg_exe(), "-y", "-i", str(src), "-vn", "-ac", str(channels), "-ar", str(sr), str(dst)])


def concat_and_mux(chunks: list[Path], audio: Path, out: Path,
                   a_start: float, a_dur: float, workdir: Path) -> None:
    """Une los trozos de vídeo (sin recodificar) y les añade el audio."""
    listfile = workdir / "chunks.txt"
    listfile.write_text(
        "".join(f"file '{c.resolve().as_posix()}'\n" for c in chunks), encoding="utf-8"
    )
    _run([
        ffmpeg_exe(), "-y",
        "-f", "concat", "-safe", "0", "-i", str(listfile),
        "-ss", f"{a_start:.3f}", "-t", f"{a_dur:.3f}", "-i", str(audio),
        "-map", "0:v", "-map", "1:a",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
        "-movflags", "+faststart", "-shortest", str(out),
    ])
