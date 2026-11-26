"""Configuración, colores y funciones de easing."""
from __future__ import annotations

from dataclasses import dataclass

Color = tuple[int, int, int]


@dataclass
class Config:
    width: int = 1920
    height: int = 1080
    fps: int = 30
    seed: int = 7
    font: str | None = None    # ruta a un .ttf/.otf
    bg: str = "#000000"        # fondo
    fg: str = "#FFFFFF"        # color del texto
    accent: str = "#FFD60A"    # color de acento (palabras clave y golpe de entrada)
    upper: bool = False        # todo en mayúsculas
    beat: bool = False         # pulso de escala al ritmo (requiere analizar el audio)
    title: str = ""
    artist: str = ""
    # style="motion": fondo animado que reacciona a la música (brillos, anillos, partículas, ecualizador)
    style: str = "kinetic"     # kinetic (fondo liso) | motion
    palette: str = "auto"      # auto (según la tonalidad), neon, sunset, ocean, forest, mono
    hue: float | None = None   # 0..1, sobrescribe el tono de la paleta
    viz: str = "bars"          # bars | ring | none
    lead: float = 0.12         # segundos que aparece un verso antes de su primera palabra (modo word)
    hold: float = 0.80         # segundos máximos que se mantiene un verso tras su última palabra
    # Cómo se revela cada verso: word (palabra a palabra), chunk (grupos de 2-3 palabras),
    # line (verso entero visible y se ilumina la palabra que suena) o auto (según la velocidad).
    reveal: str = "auto"
    line_wps: float = 2.4      # auto: a partir de estas palabras/seg el verso se muestra entero
    chunk_wps: float = 1.6     # auto: a partir de estas palabras/seg se revela por grupos
    # Transiciones entre versos permitidas (ver scene.TRANSITIONS)
    transitions: tuple = ("push_l", "push_u", "wheel", "fade", "flip", "zoom", "spin")


def hex_rgb(s: str) -> Color:
    s = s.lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def ease_out_cubic(p: float) -> float:
    return 1 - (1 - p) ** 3


def ease_out_back(p: float, k: float = 1.9) -> float:
    return 1 + (k + 1) * (p - 1) ** 3 + k * (p - 1) ** 2


def ease_out_bounce(p: float) -> float:
    n, d = 7.5625, 2.75
    if p < 1 / d:
        return n * p * p
    if p < 2 / d:
        p -= 1.5 / d
        return n * p * p + 0.75
    if p < 2.5 / d:
        p -= 2.25 / d
        return n * p * p + 0.9375
    p -= 2.625 / d
    return n * p * p + 0.984375


def mix(a: Color, b: Color, t: float) -> Color:
    t = clamp(t)
    return (int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t), int(a[2] + (b[2] - a[2]) * t))


# ------------------------------------------------------------------ paletas (estilo motion)

@dataclass
class Palette:
    bg_top: Color
    bg_bottom: Color
    accents: list
    highlight: Color


# nombre -> (tono base, esquema)
PRESETS = {
    "neon": (0.83, "triad"),
    "sunset": (0.02, "analog"),
    "ocean": (0.55, "analog"),
    "forest": (0.33, "analog"),
    "mono": (0.60, "mono"),
}
_OFFSETS = {
    "triad": [0.0, 0.33, 0.66, 0.10],
    "analog": [0.0, 0.07, -0.07, 0.14],
    "mono": [0.0, 0.02, -0.02, 0.04],
}


def _hsv(h: float, s: float, v: float) -> Color:
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def make_palette(name: str, auto_hue: float, hue: float | None = None) -> Palette:
    """auto_hue sale de la tonalidad de la canción; name/hue lo sobrescriben."""
    base, scheme = PRESETS.get(name, (auto_hue, "triad"))
    if hue is not None:
        base = hue
    offs = _OFFSETS[scheme]
    sat = 0.25 if scheme == "mono" else 0.75
    return Palette(
        bg_top=_hsv(base, 0.65, 0.17),
        bg_bottom=_hsv(base + 0.08, 0.75, 0.05),
        accents=[_hsv(base + o, sat, 1.0) for o in offs],
        highlight=_hsv(base + offs[1], 0.55 if scheme != "mono" else 0.15, 1.0),
    )
