"""Línea de comandos: canción -> videolyrics."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from .style import Config

RESOLUTIONS = {"480p": (854, 480), "720p": (1280, 720), "1080p": (1920, 1080), "1440p": (2560, 1440)}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="videolyrics", description="Genera un videolyrics de tipografía cinética desde una canción.")
    p.add_argument("audio", type=Path, help="canción (mp3, wav, flac, m4a...)")
    p.add_argument("-o", "--output", type=Path, help="vídeo de salida (por defecto output/<nombre>.mp4)")
    p.add_argument("--lyrics", type=Path, help="archivo .txt con la letra (una línea de verso por línea)")
    p.add_argument("--language", help="idioma de la letra, p. ej. es, en (por defecto autodetecta)")
    p.add_argument("--model", default="small", help="modelo Whisper: tiny, base, small, medium, large-v3")
    p.add_argument("--retranscribe", action="store_true", help="vuelve a ejecutar Whisper (ignora la caché)")
    p.add_argument("--realign", action="store_true", help="rehace lyrics.json con la transcripción en caché (sin Whisper)")
    p.add_argument("--fill-gaps", action="store_true",
                   help="con la transcripción en caché, reintenta los tramos de letra que Whisper omitió")
    p.add_argument("--lyrics-only", action="store_true", help="solo genera el JSON de tiempos y termina")
    p.add_argument("--title", help="título que se muestra al inicio (por defecto el nombre del archivo)")
    p.add_argument("--artist", default="", help="artista")
    p.add_argument("--res", default="1080p", choices=RESOLUTIONS, help="resolución")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--style", default="kinetic", choices=["kinetic", "motion"],
                   help="kinetic: fondo liso; motion: fondo animado que reacciona a la música "
                        "(brillos, anillos al ritmo, partículas y ecualizador)")
    p.add_argument("--palette", default="auto",
                   help="estilo motion: auto (según la tonalidad), neon, sunset, ocean, forest, mono")
    p.add_argument("--hue", type=float, help="estilo motion: tono 0..1 para la paleta")
    p.add_argument("--viz", default="bars", choices=["bars", "ring", "none"],
                   help="estilo motion: visualizador de audio")
    p.add_argument("--bg", default="#000000", help="color de fondo (hex; estilo kinetic)")
    p.add_argument("--fg", default="#FFFFFF", help="color del texto (hex)")
    p.add_argument("--accent", default="#FFD60A", help="color de acento: palabras clave y golpe de entrada (hex)")
    p.add_argument("--upper", action="store_true", help="todo el texto en mayúsculas")
    p.add_argument("--reveal", default="auto", choices=["auto", "word", "chunk", "line"],
                   help="cómo aparece cada verso: palabra a palabra, por grupos, entero con la palabra "
                        "que suena iluminada, o auto (según la velocidad de cada verso)")
    p.add_argument("--line-wps", type=float, default=2.4,
                   help="auto: palabras/seg a partir de las cuales el verso se muestra entero")
    p.add_argument("--transitions", default="all",
                   help="transiciones entre versos separadas por comas: push_l, push_u, wheel, fade, "
                        "flip, zoom, spin (por defecto todas)")
    p.add_argument("--beat", action="store_true", help="el texto pulsa al ritmo (analiza el audio; algo más lento)")
    p.add_argument("--font", help="ruta a una fuente .ttf/.otf")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--start", type=float, default=0.0, help="segundo desde el que renderizar")
    p.add_argument("--duration", type=float, help="segundos a renderizar (útil para pruebas)")
    p.add_argument("--frame", type=float, help="guarda solo un PNG de ese segundo y termina")
    p.add_argument("--workers", type=int, default=min(8, max(1, (os.cpu_count() or 2) - 1)),
                   help="procesos en paralelo (cada uno usa ~0,8 GB a 1080p)")
    p.add_argument("--crf", type=int, default=18, help="calidad x264 (menor = mejor)")
    p.add_argument("--preset", default="fast", help="preset x264")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.audio.exists():
        raise SystemExit(f"No existe el archivo: {args.audio}")

    from . import lyrics as ly
    from .ffmpeg_utils import decode_to_wav

    stem = args.audio.stem
    workdir = Path("output") / stem
    workdir.mkdir(parents=True, exist_ok=True)
    out = args.output or Path("output") / f"{stem}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    wav = workdir / "audio.wav"
    if not wav.exists() or wav.stat().st_mtime < args.audio.stat().st_mtime:
        print("Decodificando audio...")
        decode_to_wav(args.audio, wav)

    # ---- letra con tiempos (se cachea en JSON para poder editarla a mano)
    json_path = workdir / "lyrics.json"
    asr_path = workdir / "asr.json"
    if json_path.exists() and not (args.retranscribe or args.realign):
        print(f"Usando letra en caché: {json_path}")
        lines = ly.load_lines(json_path)
    else:
        tr = None
        fresh = args.retranscribe or not asr_path.exists()
        if fresh:
            print(f"Transcribiendo con Whisper ({args.model}); la primera vez descarga el modelo...")
            tr = ly.Transcriber(str(wav), args.language, args.model)
            words = tr.run()
            ly.save_words(words, asr_path)
        else:
            print(f"Usando transcripción en caché: {asr_path}")
            words = ly.load_words(asr_path)
        if args.lyrics:
            import soundfile as sf
            duration = sf.info(str(wav)).duration
            text = args.lyrics.read_text(encoding="utf-8")
            lines, gaps = ly.align_lyrics(text, words, duration)
            if gaps and (fresh or args.fill_gaps):
                # Whisper suele omitir tramos enteros de canciones: se reintentan solo esos
                tr = tr or ly.Transcriber(str(wav), args.language, args.model)
                for g in gaps:
                    print(f"  hueco sin reconocer {g.lo:.1f}s-{g.hi:.1f}s ({g.n_words} palabras): reintentando...")
                    extra = tr.run(clip=(max(0.0, g.lo - 0.5), min(duration, g.hi + 0.5)),
                                   prompt=g.text[:500], lenient=True)
                    print(f"    recuperadas {len(extra)} palabras")
                    words = ly.merge_words(words, extra, g.lo, g.hi)
                ly.save_words(words, asr_path)
                lines, gaps = ly.align_lyrics(text, words, duration)
            for g in gaps:
                print(f"  AVISO: {g.n_words} palabras sin reconocer entre {g.lo:.1f}s y {g.hi:.1f}s "
                      "(sus tiempos son estimados; corrígelos en lyrics.json)")
        else:
            lines = ly.group_words(words)
        lines = ly.clean_timings(lines)
        ly.save_lines(lines, json_path)
        print(f"Letra guardada en {json_path} ({len(lines)} líneas). Puedes corregirla a mano.")
    if args.lyrics_only:
        return
    if not lines:
        raise SystemExit("No se detectó letra. Prueba con --lyrics o un modelo mayor (--model medium).")

    # ---- render
    import math

    import soundfile as sf

    from .render import render_video
    w, h = RESOLUTIONS[args.res]
    from .scene import TRANSITIONS
    trans = TRANSITIONS if args.transitions == "all" else tuple(
        t.strip() for t in args.transitions.split(",") if t.strip())
    bad = [t for t in trans if t not in TRANSITIONS]
    if bad:
        raise SystemExit(f"Transiciones desconocidas: {', '.join(bad)}. Disponibles: {', '.join(TRANSITIONS)}")
    cfg = Config(width=w, height=h, fps=args.fps, seed=args.seed, font=args.font,
                 bg=args.bg, fg=args.fg, accent=args.accent, upper=args.upper, beat=args.beat,
                 reveal=args.reveal, line_wps=args.line_wps, transitions=trans,
                 style=args.style, palette=args.palette, hue=args.hue, viz=args.viz,
                 title=args.title if args.title is not None else stem, artist=args.artist)
    duration = sf.info(str(wav)).duration
    n_frames = int(math.ceil(duration * cfg.fps))
    feats = None
    if cfg.beat or cfg.style == "motion":
        from .audio import analyze
        print("Analizando audio (ritmo, espectro, tonalidad)...")
        feats = analyze(str(wav), cfg.fps)
        print(f"  tempo ~{feats.tempo:.0f} BPM")

    if args.frame is not None:
        from PIL import Image
        from .scene import Renderer
        r = Renderer(cfg, lines, n_frames, feats)
        png = out.with_suffix(".png")
        px = r.render(int(args.frame * cfg.fps))
        Image.fromarray(px[..., [2, 1, 0, 3]] if r.pix_fmt == "bgra" else px).save(png)
        print(f"Frame guardado en {png}")
        return

    render_video(cfg, lines, n_frames, feats, wav, out, workdir, args.start, args.duration,
                 args.workers, crf=args.crf, preset=args.preset)
    print(f"Listo: {out}")
