"""Tipografía cinética con Skia. Cada frame es función pura del tiempo,
así que se puede renderizar en paralelo y en cualquier orden."""
from __future__ import annotations

import math
import os
import random
import re
from dataclasses import dataclass

import numpy as np
import skia

from .audio import Features
from .background import MotionBackground
from .lyrics import Line
from .style import (Config, clamp, ease_out_back, ease_out_bounce, ease_out_cubic,
                    hex_rgb, make_palette, mix)

# Animaciones de entrada de una palabra (o grupo de palabras)
ENTERS = ("pop", "rise", "drop", "slide_l", "slide_r", "spin", "wipe", "letters")
QUICK_ENTERS = ("pop", "rise", "drop", "slide_l", "slide_r")   # legibles aunque duren poco

# Transiciones entre un verso que sale y el siguiente que entra.
#   push_l  el verso sale empujado hacia un lado y el siguiente entra por el otro
#   push_u  igual, en vertical
#   wheel   giran como un carrusel alrededor de un punto bajo la pantalla
#   fade    fundido cruzado
#   flip    el verso se voltea (como una tarjeta) y entra el siguiente
#   zoom    la cámara atraviesa el verso y el siguiente llega desde lejos
#   spin    el verso gira y se encoge; el siguiente entra girando
TRANSITIONS = ("push_l", "push_u", "wheel", "fade", "flip", "zoom", "spin")
SEQUENTIAL = {"flip"}      # primero sale uno y luego entra el otro (solo flip: los demás se solapan)
OVERSHOOT = {"push_l", "push_u", "wheel", "flip", "spin"}   # entran con un pequeño rebote

ENTER_SLOW = 0.38
ENTER_QUICK = 0.22
TR_SLOW = 0.34
TR_FAST = 0.24
LINE_LEAD = 0.30      # en modo line, el verso se ve entero antes de cantarse

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\seguibl.ttf",      # Segoe UI Black
    r"C:\Windows\Fonts\bahnschrift.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def C(rgb, a: float = 1.0) -> int:
    return skia.Color(int(rgb[0]), int(rgb[1]), int(rgb[2]), int(255 * clamp(a)))


def _typeface(path: str | None) -> skia.Typeface:
    for p in ([path] if path else []) + _FONT_CANDIDATES:
        if p and os.path.exists(p):
            tf = skia.Typeface.MakeFromFile(p)
            if tf:
                return tf
    return skia.Typeface.MakeDefault()


@dataclass
class Item:
    text: str
    x: float            # centro de la palabra, relativo al centro de pantalla
    y: float            # baseline
    size: float
    w: float
    start: float        # cuándo se canta
    end: float
    accent: bool
    row: int
    idx: int
    enter: str          # animación de entrada ("none" si la lleva el verso entero)
    enter_dur: float
    t_vis: float        # cuándo empieza a verse
    dim: float          # opacidad antes de cantarse
    vis_end: float | None = None   # solo "huge": la palabra se retira cuando entra la siguiente


@dataclass
class Group:
    items: list[Item]
    layout: str
    reveal: str
    tilt: float
    in_t: float
    out_t: float
    tin: str            # transición con la que entra el verso
    tout: str           # transición con la que sale
    sgn_in: int
    sgn_out: int
    tr_in: float
    tr_out: float


class Renderer:
    def __init__(self, cfg: Config, lines: list[Line], n_frames: int,
                 feats: Features | None = None):
        self.cfg, self.n_frames = cfg, n_frames
        self.beat_env = feats.beat_env if feats is not None and (cfg.beat or cfg.style == "motion") else None
        self.W, self.H = cfg.width, cfg.height
        self.buf = np.zeros((self.H, self.W, 4), dtype=np.uint8)
        # formato nativo + alfa premultiplicado: cualquier otra combinación es muy lenta
        self.surface = skia.Surface(self.buf, colorType=skia.kN32_ColorType,
                                    alphaType=skia.kPremul_AlphaType)
        self.pix_fmt = "bgra" if skia.ColorType.kN32_ColorType == skia.kBGRA_8888_ColorType else "rgba"
        self.typeface = _typeface(cfg.font)
        self.fonts: dict[float, skia.Font] = {}
        self.bg, self.fg, self.accent = hex_rgb(cfg.bg), hex_rgb(cfg.fg), hex_rgb(cfg.accent)
        self.paint = skia.Paint(AntiAlias=True)
        self.motion = None
        if cfg.style == "motion":
            if feats is None:
                raise ValueError("El estilo motion necesita el análisis de audio")
            pal = make_palette(cfg.palette, feats.key_hue, cfg.hue)
            self.motion = MotionBackground(cfg, feats, pal)
            self.accent = pal.highlight
        self.rng = random.Random(cfg.seed)
        self.groups = self._build([ln for ln in lines if ln.words])
        self.first_in = self.groups[0].in_t if self.groups else 0.0

    # ------------------------------------------------------------ medidas

    def font(self, size: float) -> skia.Font:
        k = round(size * 2) / 2
        f = self.fonts.get(k)
        if f is None:
            f = skia.Font(self.typeface, k)
            f.setSubpixel(True)
            self.fonts[k] = f
        return f

    def measure(self, text: str, size: float) -> float:
        return self.font(size).measureText(text)

    # ------------------------------------------------------------ composición

    @staticmethod
    def _emphasis(texts: list[str]) -> set[int]:
        """Palabras clave: la más larga y la última."""
        n = len(texts)
        norm = [re.sub(r"\W", "", t) for t in texts]
        emph: set[int] = set()
        longest = max(range(n), key=lambda i: len(norm[i]))
        if len(norm[longest]) >= 5:
            emph.add(longest)
        if len(norm[-1]) >= 4:
            emph.add(n - 1)
        return emph

    @staticmethod
    def _chunks(texts: list[str]) -> list[list[int]]:
        """Grupos de 2-3 palabras que se leen juntas (se cortan tras la puntuación)."""
        chunks: list[list[int]] = []
        cur: list[int] = []
        for k, t in enumerate(texts):
            cur.append(k)
            if len(cur) >= 3 or (t[-1] in ",.;:!?" and len(cur) >= 2):
                chunks.append(cur)
                cur = []
        if cur:
            if chunks and len(cur) == 1:
                chunks[-1] += cur
            else:
                chunks.append(cur)
        return chunks

    @staticmethod
    def _wrap(widths: list[float], space: float, maxw: float) -> list[list[int]]:
        rows: list[list[int]] = [[]]
        cur = 0.0
        for i, w in enumerate(widths):
            add = w + (space if rows[-1] else 0)
            if rows[-1] and cur + add > maxw:
                rows.append([])
                cur, add = 0.0, w
            rows[-1].append(i)
            cur += add
        if len(rows) == 2:  # equilibrar las dos filas
            best, best_cost = rows, 1e9
            for k in range(1, len(widths)):
                a, b = list(range(k)), list(range(k, len(widths)))
                cost = max(sum(widths[i] for i in r) + space * (len(r) - 1) for r in (a, b))
                if cost < best_cost:
                    best, best_cost = [a, b], cost
            rows = best
        return rows

    def _place(self, texts: list[str], emph: set[int], kind: str) -> list[dict]:
        """flow (párrafo), stack (una palabra por fila) o steps (escalera)."""
        W, H, rng = self.W, self.H, self.rng
        n = len(texts)
        one_per_row = kind == "stack"
        S = {"stack": 0.21, "steps": 0.15, "flow": 0.155}[kind] * H
        maxw = (0.62 if kind == "steps" else 0.84) * W
        indent = 0.09 * W if kind == "steps" else 0.0
        align = rng.choice(["center", "left", "right"]) if kind == "stack" else (
            "left" if kind == "steps" else "center")
        em = 1.25 if kind == "stack" else 1.3
        mult = [em if i in emph else 1.0 for i in range(n)]

        while True:
            widths = [self.measure(t, S * m) for t, m in zip(texts, mult)]
            space = self.measure(" ", S)
            rows = [[i] for i in range(n)] if one_per_row else self._wrap(widths, space, maxw)
            row_w = [sum(widths[i] for i in r) + space * (len(r) - 1) for r in rows]
            row_h = [max(S * mult[i] for i in r) * 1.1 for r in rows]
            total_h = sum(row_h)
            if (max(row_w) + indent * (len(rows) - 1) <= maxw * 1.02 and total_h <= H * 0.8) \
                    or S < H * 0.04:
                break
            S *= 0.93

        margin = 0.08 * W
        starts = []
        for ri in range(len(rows)):
            if align == "center":
                starts.append(-row_w[ri] / 2)
            elif align == "left":
                starts.append(-W / 2 + margin + ri * indent)
            else:
                starts.append(W / 2 - margin - row_w[ri])
        if kind == "steps":  # centrar el bloque completo
            lo, hi = min(starts), max(s + w for s, w in zip(starts, row_w))
            starts = [s - (lo + hi) / 2 for s in starts]

        out: list[dict] = [{}] * n
        y = -total_h / 2
        for ri, r in enumerate(rows):
            base = y + row_h[ri] * 0.82
            x = starts[ri]
            for i in r:
                out[i] = dict(x=x + widths[i] / 2, y=base, size=S * mult[i], w=widths[i], row=ri)
                x += widths[i] + space
            y += row_h[ri]
        return out

    def _place_huge(self, texts: list[str]) -> list[dict]:
        out = []
        for t in texts:
            S = 0.30 * self.H
            while self.measure(t, S) > 0.86 * self.W and S > self.H * 0.04:
                S *= 0.93
            out.append(dict(x=0.0, y=S * 0.36, size=S, w=self.measure(t, S), row=0))
        return out

    def _choose(self, options: list[tuple[str, float]], prev: str | None) -> str:
        opts = [o for o in options if o[0] != prev] or options
        return self.rng.choices([o[0] for o in opts], [o[1] for o in opts])[0]

    # ------------------------------------------------------------ construcción

    def _build_items(self, line: Line, texts: list[str], reveal: str, layout: str,
                     wps: float) -> list[Item]:
        rng, n = self.rng, len(texts)
        emph = self._emphasis(texts)
        pos = self._place_huge(texts) if layout == "huge" else self._place(texts, emph, layout)

        if reveal == "word":
            pool = [e for e in ENTERS if not (e == "letters" and wps >= 1.8)]
            enter = self._choose([(e, 1.0) for e in pool], self._prev_enter)
            self._prev_enter = enter
        elif reveal == "chunk":
            enter = self._choose([(e, 1.0) for e in QUICK_ENTERS], self._prev_enter)
            self._prev_enter = enter
        else:
            enter = "none"
        chunks = self._chunks(texts) if reveal == "chunk" else []
        chunk_of = {k: (ci, j) for ci, c in enumerate(chunks) for j, k in enumerate(c)}

        items = []
        for k, (w, p) in enumerate(zip(line.words, pos)):
            en, dur, t_vis, dim = enter, ENTER_SLOW, w.start - 0.05, 1.0
            if layout == "huge":
                en = rng.choice([e for e in ENTERS if e != "letters" or len(texts[k]) <= 9])
            if reveal == "chunk":
                ci, j = chunk_of[k]
                t_vis = line.words[chunks[ci][0]].start - 0.12 + j * 0.04
                dur, dim = ENTER_QUICK, 0.6
            elif reveal == "line":
                dur, t_vis, dim = 1.0, -1e9, 0.30
            vis_end = line.words[k + 1].start if layout == "huge" and k + 1 < n else None
            items.append(Item(texts[k], p["x"], p["y"], p["size"], p["w"], w.start, w.end,
                              k in emph, p["row"], k, en, dur, t_vis, dim, vis_end))
        return items

    def _build(self, lines: list[Line]) -> list[Group]:
        cfg, rng = self.cfg, self.rng
        pool = [t for t in cfg.transitions if t in TRANSITIONS] or list(TRANSITIONS)
        self._prev_enter = None

        # 1) composición y revelado de cada verso
        metas = []
        prev_layout = None
        for line in lines:
            texts = [w.text.upper() if cfg.upper else w.text for w in line.words]
            n = len(texts)
            wps = n / max(0.3, line.end - line.start)
            if cfg.reveal == "auto":
                reveal = "line" if wps >= cfg.line_wps else "chunk" if wps >= cfg.chunk_wps else "word"
            else:
                reveal = cfg.reveal
            layouts = [("flow", 3.0)]
            if 2 <= n <= 5:
                layouts.append(("stack", 2.5))
            if 3 <= n <= 9:
                layouts.append(("steps", 2.0))
            if reveal == "word" and n <= 6 and wps < 2.4:
                layouts.append(("huge", 1.5))     # una palabra cada vez: solo si el verso es lento
            layout = self._choose(layouts, prev_layout)
            prev_layout = layout
            if layout == "huge":
                reveal = "word"
            tilt = rng.uniform(-4, 4) if layout in ("stack", "steps") and rng.random() < 0.6 else 0.0
            metas.append((line, reveal, layout, tilt, self._build_items(line, texts, reveal, layout, wps)))

        # 2) transiciones entre versos consecutivos
        m = len(metas)
        kinds, sgns, trs = [], [], []
        prev_kind = None
        for i in range(m):
            line, reveal = metas[i][0], metas[i][1]
            nxt = metas[i + 1][0] if i + 1 < m else None
            tight = nxt is not None and nxt.start - line.end < 0.5
            fast = tight or reveal != "word"
            allowed = [t for t in pool if not (tight and t in SEQUENTIAL)] or ["fade"]
            kind = self._choose([(t, 1.0) for t in allowed], prev_kind)
            prev_kind = kind
            kinds.append(kind)
            sgns.append(rng.choice([-1, 1]))
            trs.append(TR_FAST if fast else TR_SLOW)

        # 3) horarios: el verso anterior sale mientras entra el siguiente
        groups: list[Group] = []
        out_prev = None
        for i, (line, reveal, layout, tilt, items) in enumerate(metas):
            lead = LINE_LEAD if reveal == "line" else cfg.lead
            seq_in = i > 0 and kinds[i - 1] in SEQUENTIAL
            desired = line.start - lead - (trs[i - 1] / 2 if seq_in else 0.0)
            tr_in = trs[i - 1] if i > 0 else TR_SLOW
            tin = kinds[i - 1] if i > 0 else self._choose([(t, 1.0) for t in pool], None)
            sgn_in = sgns[i - 1] if i > 0 else rng.choice([-1, 1])
            in_t = desired if out_prev is None else max(desired, out_prev - trs[i - 1])
            in_t = min(in_t, line.start - 0.03)

            tr_out = trs[i]
            if i + 1 < m:
                nl, nreveal = metas[i + 1][0], metas[i + 1][1]
                nlead = LINE_LEAD if nreveal == "line" else cfg.lead
                n_desired = nl.start - nlead - (tr_out / 2 if kinds[i] in SEQUENTIAL else 0.0)
                out_t = min(line.end + cfg.hold, n_desired + tr_out)
                out_t = max(out_t, line.end + 0.6 * tr_out)   # la última palabra se lee entera
            else:
                out_t = line.end + cfg.hold
            out_prev = out_t
            groups.append(Group(items, layout, reveal, tilt, in_t, out_t, tin, kinds[i], sgn_in,
                                sgns[i], tr_in, tr_out))
        return groups

    # ------------------------------------------------------------ frame

    def render(self, frame: int) -> np.ndarray:
        canvas = self.surface.getCanvas()
        t = frame / self.cfg.fps
        canvas.clear(C(self.bg))
        if self.motion is not None:
            self.motion.draw(canvas, frame)
        pulse = 1.0
        if self.beat_env is not None:
            pulse += 0.025 * float(self.beat_env[min(frame, len(self.beat_env) - 1)])
        self._title(canvas, t)
        self._lyrics(canvas, t, pulse)
        return self.buf

    def _phrase(self, canvas: skia.Canvas, kind: str, sgn: int, h: float, entering: bool) -> float:
        """Aplica la transformación de una transición al verso entero y devuelve su opacidad.
        h: 0 = verso colocado, 1 = verso fuera. Puede ser negativo (rebote al entrar)."""
        W, H = self.W, self.H
        side = (1 if entering else -1) * sgn
        hc = clamp(h)
        if kind == "push_l":
            canvas.translate(side * W * h, 0)
            return 1.0
        if kind == "push_u":
            canvas.translate(0, side * H * h)
            return 1.0
        if kind == "wheel":
            pivot = H * 1.3
            canvas.translate(0, pivot)
            canvas.rotate(side * 38 * h)
            canvas.translate(0, -pivot)
            return 1 - hc
        if kind == "flip":
            canvas.scale(1, max(0.02, 1 - h))
            return 1 - hc * hc
        if kind == "zoom":
            s = 1 - 0.45 * h if entering else 1 + 1.6 * h
            canvas.scale(s, s)
            return 1 - hc
        if kind == "spin":
            canvas.rotate(side * 110 * h)
            s = 1 - 0.6 * hc
            canvas.scale(s, s)
            return 1 - 0.9 * hc
        s = 1 - 0.05 * hc    # fade
        canvas.scale(s, s)
        return 1 - hc

    def _lyrics(self, canvas: skia.Canvas, t: float, pulse: float) -> None:
        for g in self.groups:
            if t < g.in_t or t > g.out_t:
                continue
            canvas.save()
            canvas.translate(self.W / 2, self.H / 2)
            ga = 1.0
            # entrada: los tipos secuenciales usan la segunda mitad de la ventana
            if t < g.in_t + g.tr_in:
                if g.tin in SEQUENTIAL:
                    p = clamp((t - g.in_t - g.tr_in / 2) / (g.tr_in / 2))
                else:
                    p = clamp((t - g.in_t) / g.tr_in)
                e = ease_out_back(p, 1.3) if g.tin in OVERSHOOT else ease_out_cubic(p)
                ga *= self._phrase(canvas, g.tin, g.sgn_in, 1 - e, True)
            # salida: los secuenciales usan la primera mitad
            if t > g.out_t - g.tr_out:
                span = g.tr_out / 2 if g.tout in SEQUENTIAL else g.tr_out
                p = clamp((t - (g.out_t - g.tr_out)) / span)
                ga *= self._phrase(canvas, g.tout, g.sgn_out, p * p, False)
            if ga <= 0.01:
                canvas.restore()
                continue
            canvas.rotate(g.tilt)
            life = clamp((t - g.in_t) / max(0.1, g.out_t - g.in_t))
            s = (1 + 0.05 * life) * pulse      # deriva lenta de cámara durante el verso
            canvas.scale(s, s)
            for it in g.items:
                self._item(canvas, it, t, ga)
            canvas.restore()

    def _item(self, canvas: skia.Canvas, it: Item, t: float, ga: float) -> None:
        W, H = self.W, self.H
        if t < it.t_vis or (it.vis_end is not None and t >= it.vis_end):
            return
        p = clamp((t - it.t_vis) / it.enter_dur)
        dx = dy = rot = 0.0
        sc, alpha, wipe, letters = 1.0, ga, 1.0, False

        a = it.enter
        if a == "pop":
            sc, alpha = 0.3 + 0.7 * ease_out_back(p), alpha * clamp(p * 4)
        elif a == "rise":
            e = ease_out_cubic(p)
            dy, alpha = (1 - e) * H * 0.09, alpha * e
        elif a == "drop":
            dy, alpha = -(1 - ease_out_bounce(p)) * H * 0.14, alpha * clamp(p * 4)
        elif a in ("slide_l", "slide_r"):
            e = ease_out_cubic(p)
            dx, alpha = (1 - e) * W * 0.18 * (-1 if a == "slide_l" else 1), alpha * e
        elif a == "spin":
            e = ease_out_back(p)
            rot, sc, alpha = -28 * (1 - e), 0.4 + 0.6 * e, alpha * clamp(p * 4)
        elif a == "wipe":
            wipe = ease_out_cubic(p)
        elif a == "letters":
            letters = True

        ts = t - it.start
        if ts < 0:
            alpha *= it.dim                                  # aún no se canta
            color = self.accent if it.accent else self.fg
        else:
            alpha *= it.dim + (1 - it.dim) * clamp(ts / 0.08)
            sc *= 1 + 0.12 * math.exp(-ts / 0.10)            # golpe al cantarse
            color = self.accent if it.accent else mix(self.accent, self.fg, ts / 0.30)
        if alpha <= 0.01 or sc <= 0.01:
            return

        font = self.font(it.size)
        base = it.size * 0.35
        canvas.save()
        canvas.translate(it.x + dx, it.y + dy - base)
        canvas.rotate(rot)
        canvas.scale(sc, sc)
        if wipe < 1.0:
            canvas.clipRect(skia.Rect.MakeLTRB(-it.w / 2 - 6, -it.size, -it.w / 2 + it.w * wipe, it.size))
        if self.motion is not None and not letters:
            canvas.drawString(it.text, -it.w / 2, base + it.size * 0.04, font, skia.Paint(
                AntiAlias=True, Color=C((0, 0, 0), 0.4 * alpha)))
            if 0 <= ts < 0.5:
                canvas.drawString(it.text, -it.w / 2, base, font, skia.Paint(
                    AntiAlias=True, Color=C(self.accent, 0.8 * alpha * (1 - ts / 0.5)),
                    MaskFilter=skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, it.size * 0.11)))
        if letters:
            for ci, ch in enumerate(it.text):
                pc = clamp((t - it.t_vis - ci * 0.03) / 0.28)
                if pc <= 0:
                    continue
                e = ease_out_back(pc, 2.4)
                self.paint.setColor(C(color, alpha * clamp(pc * 3)))
                canvas.drawString(ch, -it.w / 2 + font.measureText(it.text[:ci]),
                                  base + (1 - e) * it.size * 0.45, font, self.paint)
        else:
            self.paint.setColor(C(color, alpha))
            canvas.drawString(it.text, -it.w / 2, base, font, self.paint)
        canvas.restore()

    def _title(self, canvas: skia.Canvas, t: float) -> None:
        cfg = self.cfg
        if not cfg.title or self.first_in < 2.5:
            return
        a_in = ease_out_cubic(clamp((t - 0.4) / 0.9))
        a = a_in * (1 - clamp((t - (self.first_in - 0.7)) / 0.5))
        if a <= 0:
            return
        size = self.H * 0.11
        while self.measure(cfg.title, size) > self.W * 0.85 and size > 20:
            size *= 0.92
        w = self.measure(cfg.title, size)
        y = self.H / 2 + (1 - a_in) * 24
        self.paint.setColor(C(self.fg, a))
        canvas.drawString(cfg.title, self.W / 2 - w / 2, y, self.font(size), self.paint)
        if cfg.artist:
            s2 = self.H * 0.045
            w2 = self.measure(cfg.artist.upper(), s2)
            self.paint.setColor(C(self.accent, a))
            canvas.drawString(cfg.artist.upper(), self.W / 2 - w2 / 2, y + self.H * 0.09,
                              self.font(s2), self.paint)
