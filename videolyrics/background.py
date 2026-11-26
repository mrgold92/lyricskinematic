"""Fondo animado del estilo "motion": brillos que pulsan con los graves, anillos en cada beat,
partículas, ecualizador, viñeta y destello de beat. Todo es función del frame.

Notas de rendimiento (Skia raster en CPU): los degradados a pantalla completa y el modo aditivo
sobre imágenes grandes son muy lentos, así que lo estático se cachea como imagen, los brillos se
dibujan a 1/4 de resolución y el destello usa una imagen blanca con alfa."""
from __future__ import annotations

import math
import random

import numpy as np
import skia

from .audio import N_BARS, Features
from .style import Config, Palette, clamp


def _c(rgb, a: float = 1.0) -> int:
    return skia.Color(int(rgb[0]), int(rgb[1]), int(rgb[2]), int(255 * clamp(a)))


class MotionBackground:
    def __init__(self, cfg: Config, feats: Features, pal: Palette):
        self.W, self.H, self.fps = cfg.width, cfg.height, cfg.fps
        self.feats, self.pal, self.viz = feats, pal, cfg.viz
        W, H = self.W, self.H
        s = H / 1080.0
        rng = random.Random(cfg.seed + 1)

        self.blobs = [dict(
            fx=rng.uniform(0.02, 0.07), fy=rng.uniform(0.02, 0.07),
            px=rng.uniform(0, 6.28), py=rng.uniform(0, 6.28),
            r=rng.uniform(0.28, 0.5), col=pal.accents[i % len(pal.accents)],
            ax=rng.uniform(0.25, 0.42), ay=rng.uniform(0.2, 0.38),
        ) for i in range(6)]
        self.particles = [dict(
            x=rng.uniform(0, W), y=rng.uniform(0, H),
            vy=rng.uniform(15, 80) * s, sway=rng.uniform(8, 50) * s,
            size=rng.uniform(1.5, 5.5) * s, p=rng.uniform(0, 6.28),
            tw=rng.uniform(0.6, 2.2), col=rng.choice(pal.accents),
        ) for _ in range(150)]
        self.s = s

        def layer(paint: skia.Paint) -> skia.Image:
            surf = skia.Surface(W, H)
            surf.getCanvas().drawPaint(paint)
            return surf.makeImageSnapshot()

        self.bg_image = layer(skia.Paint(Shader=skia.GradientShader.MakeLinear(
            [skia.Point(0, 0), skia.Point(0, H)], [_c(pal.bg_top), _c(pal.bg_bottom)])))
        self.vignette = layer(skia.Paint(Shader=skia.GradientShader.MakeRadial(
            skia.Point(W / 2, H / 2), math.hypot(W, H) * 0.55,
            [_c((0, 0, 0), 0), _c((0, 0, 0), 0), _c((0, 0, 0), .72)], [0, .45, 1])))
        self.white = layer(skia.Paint(Color=skia.ColorWHITE))
        a = pal.accents
        self.bar_shader = skia.GradientShader.MakeLinear(
            [skia.Point(0, 0), skia.Point(W, 0)],
            [_c(a[0], .75), _c(a[1], .75), _c(a[2], .75), _c(a[1], .75), _c(a[0], .75)],
            [0, .25, .5, .75, 1])
        self.gs = 0.25
        self.glow = skia.Surface(int(W * self.gs), int(H * self.gs))

    def draw(self, canvas: skia.Canvas, frame: int) -> None:
        W, H, ft = self.W, self.H, self.feats
        f = min(frame, ft.n_frames - 1)
        t = frame / self.fps
        bass, rms, beat = float(ft.bass[f]), float(ft.rms[f]), float(ft.beat_env[f])

        canvas.drawImage(self.bg_image, 0, 0)
        zoom = 1.07 + 0.012 * math.sin(t * 0.35) + 0.018 * beat ** 2
        rot = 1.1 * math.sin(t * 0.13)

        def camera(cv: skia.Canvas, k: float = 1.0) -> None:
            cv.translate(W * k / 2, H * k / 2)
            cv.rotate(rot)
            cv.scale(zoom, zoom)
            cv.translate(-W * k / 2, -H * k / 2)

        # brillos difusos (1/4 de resolución, con la cámara aplicada en esa capa)
        gs, gc = self.gs, self.glow.getCanvas()
        gc.clear(skia.ColorTRANSPARENT)
        gc.save()
        camera(gc, gs)
        for b in self.blobs:
            x = W * (0.5 + b["ax"] * math.sin(t * b["fx"] * 6.28 + b["px"]))
            y = H * (0.5 + b["ay"] * math.sin(t * b["fy"] * 6.28 + b["py"]))
            r = H * b["r"] * (1 + 0.35 * bass)
            alpha = 0.30 + 0.32 * rms
            shader = skia.GradientShader.MakeRadial(
                skia.Point(x * gs, y * gs), r * gs, [_c(b["col"], alpha), _c(b["col"], 0)], [0, 1])
            gc.drawCircle(x * gs, y * gs, r * gs, skia.Paint(Shader=shader, BlendMode=skia.BlendMode.kPlus))
        gc.restore()
        canvas.drawImageRect(self.glow.makeImageSnapshot(), skia.Rect.MakeWH(W, H),
                             skia.SamplingOptions(skia.FilterMode.kLinear))

        canvas.save()
        camera(canvas)
        # anillos que nacen en cada beat
        bt = ft.beat_times
        k = int(np.searchsorted(bt, t, side="right"))
        for bi in range(max(0, k - 4), k):
            age = t - float(bt[bi])
            if age > 1.8:
                continue
            q = age / 1.8
            canvas.drawCircle(W / 2, H / 2, H * 0.08 + q * H * 0.75, skia.Paint(
                AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=(2 + 7 * (1 - q)) * self.s,
                Color=_c(self.pal.accents[bi % 3], 0.38 * (1 - q) ** 2), BlendMode=skia.BlendMode.kPlus))
        # partículas
        phase, high = float(ft.phase[f]), float(ft.high[f])
        p = skia.Paint(AntiAlias=True, BlendMode=skia.BlendMode.kPlus)
        for q in self.particles:
            y = (q["y"] - q["vy"] * phase) % H
            x = q["x"] + q["sway"] * math.sin(phase * 0.7 + q["p"])
            tw = 0.5 + 0.5 * math.sin(t * q["tw"] + q["p"])
            p.setColor(_c(q["col"], (0.15 + 0.5 * tw) * (0.6 + 0.8 * high)))
            canvas.drawCircle(x, y, q["size"] * (1 + 0.4 * beat), p)
        canvas.restore()

        if self.viz == "bars":
            self._bars(canvas, f)
        elif self.viz == "ring":
            self._ring(canvas, f, t)
        canvas.drawImage(self.vignette, 0, 0)
        if beat > 0.02:
            canvas.drawImage(self.white, 0, 0, skia.SamplingOptions(),
                             skia.Paint(Alpha=int(255 * 0.05 * beat)))

    def _bars(self, canvas: skia.Canvas, f: int) -> None:
        W, H, n = self.W, self.H, N_BARS
        bars = self.feats.bars[f]
        slot = W / n
        paint = skia.Paint(AntiAlias=True, Shader=self.bar_shader)
        for i in range(n):
            k = abs(i - (n - 1) / 2) / ((n - 1) / 2)
            v = float(bars[int(k * (N_BARS - 1))])
            h = 6 + v ** 1.4 * H * 0.20
            rect = skia.Rect.MakeLTRB(i * slot + slot * 0.18, H - h, (i + 1) * slot - slot * 0.18, H + 12)
            canvas.drawRoundRect(rect, slot * 0.3, slot * 0.3, paint)

    def _ring(self, canvas: skia.Canvas, f: int, t: float) -> None:
        n, r0 = 96, self.H * 0.34 * (1 + 0.05 * float(self.feats.bass[f]))
        bars = self.feats.bars[f]
        for i in range(n):
            ang = (i / n) * 2 * math.pi + t * 0.15
            v = float(bars[int(abs(i - n / 2) / (n / 2) * (N_BARS - 1))])
            r1 = r0 + 6 * self.s + v ** 1.3 * self.H * 0.12
            c, s = math.cos(ang), math.sin(ang)
            canvas.drawLine(
                self.W / 2 + c * r0, self.H / 2 + s * r0, self.W / 2 + c * r1, self.H / 2 + s * r1,
                skia.Paint(AntiAlias=True, StrokeWidth=5 * self.s, StrokeCap=skia.Paint.kRound_Cap,
                           Color=_c(self.pal.accents[i % 3], 0.55), BlendMode=skia.BlendMode.kPlus))
