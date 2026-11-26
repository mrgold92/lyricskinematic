"""Render paralelo: cada proceso codifica su propio trozo de vídeo con ffmpeg."""
from __future__ import annotations

import multiprocessing as mp
import subprocess
from pathlib import Path

from tqdm import tqdm

from .audio import Features
from .ffmpeg_utils import concat_and_mux, ffmpeg_exe
from .lyrics import Line
from .scene import Renderer
from .style import Config

CHUNK_SECONDS = 8
_R: Renderer | None = None


def _init(cfg: Config, lines: list[Line], n_frames: int, feats: Features | None) -> None:
    global _R
    _R = Renderer(cfg, lines, n_frames, feats)


def _tail(log: Path, n: int = 1500) -> str:
    return log.read_text(errors="replace")[-n:]


def _encode_chunk(job: tuple) -> tuple[int, int]:
    idx, f0, f1, path, crf, preset = job
    cfg = _R.cfg
    cmd = [
        ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", _R.pix_fmt, "-s", f"{cfg.width}x{cfg.height}",
        "-r", str(cfg.fps), "-i", "-",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-threads", "2", "-g", str(cfg.fps * 2), str(path),
    ]
    log = Path(path).with_suffix(".log")
    with open(log, "wb") as errf:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=errf)
        try:
            for f in range(f0, f1):
                proc.stdin.write(_R.render(f).tobytes())
            proc.stdin.close()
        except OSError as e:  # ffmpeg murió: en Windows se ve como EINVAL / BrokenPipe
            proc.wait()
            raise RuntimeError(f"ffmpeg falló en el trozo {idx} ({e}):\n{_tail(log)}")
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg falló en el trozo {idx}:\n{_tail(log)}")
    log.unlink(missing_ok=True)
    return idx, f1 - f0


def render_video(cfg: Config, lines: list[Line], n_frames: int, feats: Features | None,
                 wav: Path, out: Path, workdir: Path, start: float, duration: float | None,
                 workers: int, crf: int = 18, preset: str = "fast") -> None:
    f_start = int(start * cfg.fps)
    f_end = n_frames if duration is None else min(n_frames, f_start + int(duration * cfg.fps))
    if f_end <= f_start:
        raise ValueError("El rango pedido está fuera de la duración de la canción")

    step = CHUNK_SECONDS * cfg.fps
    jobs = []
    for i, f0 in enumerate(range(f_start, f_end, step)):
        jobs.append((i, f0, min(f0 + step, f_end), workdir / f"chunk_{i:04d}.mp4", crf, preset))

    total = f_end - f_start
    ctx = mp.get_context("spawn")
    workers = max(1, min(workers, len(jobs)))
    with ctx.Pool(workers, _init, (cfg, lines, n_frames, feats)) as pool, \
            tqdm(total=total, unit="frame", desc="Renderizando") as bar:
        for _, n in pool.imap_unordered(_encode_chunk, jobs):
            bar.update(n)

    print("Uniendo trozos y añadiendo audio...")
    concat_and_mux([j[3] for j in jobs], wav, out, f_start / cfg.fps,
                   (f_end - f_start) / cfg.fps, workdir)
    for j in jobs:
        j[3].unlink(missing_ok=True)
