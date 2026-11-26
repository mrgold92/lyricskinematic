"""Letras con tiempos por palabra: transcripción, alineación y (de)serialización."""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


@dataclass
class Word:
    text: str
    start: float
    end: float
    brk: bool = False  # salto de línea previo (inicio de segmento de Whisper)


@dataclass
class Line:
    words: list[Word]

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


# ------------------------------------------------------------------ transcripción

class Transcriber:
    """Whisper en CPU con tiempos por palabra. Mantiene el modelo cargado entre llamadas."""

    def __init__(self, audio_path: str, language: str | None, model_size: str):
        import soundfile as sf
        from faster_whisper import WhisperModel

        from .ffmpeg_utils import decode_to_wav

        # Decodificamos nosotros (16 kHz mono) para no depender de PyAV dentro de faster-whisper
        wav16 = Path(audio_path).with_name("audio_16k.wav")
        decode_to_wav(Path(audio_path), wav16, sr=16000, channels=1)
        self.audio, _ = sf.read(str(wav16), dtype="float32")
        self.language = language
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def run(self, clip: tuple[float, float] | None = None, prompt: str | None = None,
            lenient: bool = False) -> list[Word]:
        """lenient=True desactiva los filtros de "aquí no hay voz" (para rescatar tramos omitidos)."""
        kw: dict = {}
        if clip:
            kw["clip_timestamps"] = [clip[0], clip[1]]
        if prompt:
            kw["initial_prompt"] = prompt
        if lenient:
            kw.update(no_speech_threshold=None, log_prob_threshold=None, compression_ratio_threshold=None)
        segments, info = self.model.transcribe(
            self.audio, language=self.language, word_timestamps=True,
            condition_on_previous_text=False, vad_filter=False, beam_size=5, **kw)
        if not clip:
            print(f"  idioma: {info.language} (p={info.language_probability:.2f})")
        words: list[Word] = []
        for seg in segments:
            if not seg.words or (not lenient and seg.no_speech_prob > 0.7):
                continue
            for i, w in enumerate(seg.words):
                text = w.word.strip()
                if text:
                    words.append(Word(text, float(w.start), float(w.end), brk=(i == 0)))
        return words


def merge_words(asr: list[Word], extra: list[Word], lo: float, hi: float) -> list[Word]:
    """Sustituye las palabras de [lo, hi] por las recuperadas."""
    keep = [w for w in asr if not (w.start >= lo - 0.5 and w.end <= hi + 0.5)]
    new = [w for w in extra if w.start >= lo - 0.5 and w.end <= hi + 0.5]
    return sorted(keep + new, key=lambda w: w.start)


def group_words(words: list[Word], max_words: int = 7, max_chars: int = 38,
                max_gap: float = 0.9) -> list[Line]:
    """Agrupa palabras sueltas en líneas de letra legibles."""
    lines: list[Line] = []
    cur: list[Word] = []
    for w in words:
        if cur:
            gap = w.start - cur[-1].end
            chars = sum(len(x.text) + 1 for x in cur) + len(w.text)
            ends_phrase = cur[-1].text[-1] in ".,;:!?" and len(cur) >= 3
            if w.brk or gap > max_gap or len(cur) >= max_words or chars > max_chars or ends_phrase:
                lines.append(Line(cur))
                cur = []
        cur.append(w)
    if cur:
        lines.append(Line(cur))
    return lines


# ------------------------------------------------------------------ alineación

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^\w]", "", s)


@dataclass
class Gap:
    """Tramo de letra que Whisper no reconoció (probablemente lo omitió)."""
    lo: float
    hi: float
    text: str
    n_words: int


def align_lyrics(lyrics_text: str, asr: list[Word], duration: float) -> tuple[list[Line], list[Gap]]:
    """Reparte los tiempos de la transcripción sobre la letra real del usuario.

    Devuelve las líneas y los huecos largos sin reconocer, que conviene volver a transcribir."""
    gaps: list[Gap] = []
    rows = [ln.split() for ln in lyrics_text.splitlines() if ln.strip()]
    flat = [(li, tok) for li, row in enumerate(rows) for tok in row]
    n = len(flat)
    if n == 0:
        raise ValueError("El archivo de letra está vacío")

    a_norm = [_norm(w.text) for w in asr]
    l_norm = [_norm(t) for _, t in flat]
    start: list[float | None] = [None] * n
    end: list[float | None] = [None] * n
    matched = 0
    for a, b, size in SequenceMatcher(None, a_norm, l_norm, autojunk=False).get_matching_blocks():
        for k in range(size):
            if l_norm[b + k]:
                start[b + k], end[b + k] = asr[a + k].start, asr[a + k].end
                matched += 1
    print(f"  palabras de la letra reconocidas en el audio: {matched}/{n} ({100 * matched // n}%)")
    if matched < n * 0.4:
        print("  AVISO: poca coincidencia; revisa idioma/modelo o edita el JSON a mano.")

    cap = 0.7 if matched else None  # sin anclas, repartir uniformemente
    i = 0
    while i < n:
        if start[i] is not None:
            i += 1
            continue
        j = i
        while j < n and start[j] is None:
            j += 1
        lo = end[i - 1] if i > 0 else None
        hi = start[j] if j < n else None
        if lo is None:
            lo = max(0.0, hi - 0.4 * (j - i)) if hi is not None else (asr[0].start if asr else 0.0)
        if hi is None:
            hi = duration if cap is None else lo + 0.4 * (j - i)
        if j - i >= 8 and hi - lo >= 6.0:
            gaps.append(Gap(lo, hi, " ".join(flat[k][1] for k in range(i, j)), j - i))
        weights = [len(flat[k][1]) + 1 for k in range(i, j)]
        total_w = sum(weights)
        t = lo
        for k, wt in zip(range(i, j), weights):
            d = max(0.12, (hi - lo) * wt / total_w)
            if cap:
                d = min(cap, d)
            start[k], end[k] = t, t + d
            t += d
        i = j

    words = [Word(tok, start[k], end[k]) for k, (_, tok) in enumerate(flat)]  # type: ignore[arg-type]
    lines: list[Line] = []
    for li in range(len(rows)):
        lw = [w for w, (l, _) in zip(words, flat) if l == li]
        if lw:
            lines.append(Line(lw))
    return lines, gaps


# ------------------------------------------------------------------ limpieza

def clean_timings(lines: list[Line], min_dur: float = 0.12, max_dur: float = 2.5) -> list[Line]:
    """Garantiza tiempos monótonos, sin solapes y con duraciones razonables."""
    flat = [w for ln in lines for w in ln.words]
    prev_end = 0.0
    for w in flat:
        w.start = max(w.start, prev_end - 0.02)
        w.end = min(max(w.end, w.start + min_dur), w.start + max_dur)
        prev_end = w.end
    for a, b in zip(flat, flat[1:]):
        if a.end > b.start:
            a.end = max(a.start + 0.05, b.start)
    return lines


# ------------------------------------------------------------------ JSON editable

def save_lines(lines: list[Line], path: Path) -> None:
    rows = [
        "  " + json.dumps([[w.text, round(w.start, 3), round(w.end, 3)] for w in ln.words],
                          ensure_ascii=False)
        for ln in lines
    ]
    path.write_text("[\n" + ",\n".join(rows) + "\n]\n", encoding="utf-8")


def load_lines(path: Path) -> list[Line]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Line([Word(t, float(s), float(e)) for t, s, e in row]) for row in data if row]


def save_words(words: list[Word], path: Path) -> None:
    rows = ["  " + json.dumps([w.text, round(w.start, 3), round(w.end, 3), int(w.brk)], ensure_ascii=False)
            for w in words]
    path.write_text("[\n" + ",\n".join(rows) + "\n]\n", encoding="utf-8")


def load_words(path: Path) -> list[Word]:
    return [Word(t, float(s), float(e), bool(b)) for t, s, e, b in json.loads(path.read_text(encoding="utf-8"))]
