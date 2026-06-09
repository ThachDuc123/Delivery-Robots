"""Procedural 2D corridor worlds for sensor-only navigation (numpy only).

A world is a set of axis-free **wall segments** (line segments) plus a free-space
test. Corridors are generated from a random **centerline** (a poly-line that goes
straight, bends in arcs / corners, and may add dead-end niches) by offsetting it
left and right by a half-width. Because every episode draws a NEW layout
(straight / arc / L-turn / S-curve / niches), a policy that only sees a LiDAR fan
must learn to *follow the corridor* rather than memorise one map -> it
generalises to unseen maps.

No PyBullet, no rendering here -- pure geometry so it is fast enough for millions
of steps across many parallel envs.

Public API used by the env:
  World.generate(rng, style) -> builds walls + centerline + start/goal
  World.segments : list[(x1,y1,x2,y2)]
  World.centerline : list[(x,y)]   (for spawn + progress + drawing)
  World.raycast(origin, angle, max_range) -> distance
  World.segment_hits_circle(p, r) -> bool   (collision)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

Vec = Tuple[float, float]
Seg = Tuple[float, float, float, float]

STYLES = ("straight", "L_turn", "S_curve", "arc", "U_turn", "niches")


@dataclass
class World:
    half_width: float = 1.4              # corridor half-width (m) -> ~2.8 m wide
    segments: List[Seg] = field(default_factory=list)
    centerline: List[Vec] = field(default_factory=list)
    start: Vec = (0.0, 0.0)
    goal: Vec = (0.0, 0.0)
    start_heading: float = 0.0
    bounds: Tuple[float, float, float, float] = (-1, 1, -1, 1)
    style: str = "straight"

    # ------------------------------------------------------------------ #
    #  Generation
    # ------------------------------------------------------------------ #
    @staticmethod
    def generate(rng: np.random.Generator, style: Optional[str] = None,
                 half_width: Optional[float] = None) -> "World":
        style = style or rng.choice(STYLES)
        hw = half_width if half_width is not None else float(rng.uniform(1.1, 1.7))
        w = World(half_width=hw, style=style)
        cl = w._make_centerline(rng, style)
        w.centerline = cl
        w._build_walls_from_centerline(cl, rng)
        # Start / goal sit a few cells INSIDE the end-caps (never on a wall), and
        # the centerline is trimmed so progress/return target the inset points.
        inset = 3
        inset = min(inset, max(1, len(cl) // 2 - 1))
        w.start = cl[inset]
        w.goal = cl[-1 - inset]
        d = np.array(cl[inset + 1]) - np.array(cl[inset])
        w.start_heading = math.atan2(d[1], d[0])
        xs = [p[0] for p in cl]; ys = [p[1] for p in cl]
        m = hw + 2.0
        w.bounds = (min(xs) - m, max(xs) + m, min(ys) - m, max(ys) + m)
        return w

    def _make_centerline(self, rng, style) -> List[Vec]:
        """Return a poly-line (dense, ~0.3 m spacing) describing the hallway centre."""
        pts: List[Vec] = [(0.0, 0.0)]
        heading = 0.0
        step = 0.3

        def advance(n, turn_per=0.0):
            nonlocal heading
            for _ in range(n):
                heading += turn_per
                x, y = pts[-1]
                pts.append((x + step * math.cos(heading), y + step * math.sin(heading)))

        if style == "straight":
            advance(int(rng.uniform(28, 60)))
        elif style == "L_turn":
            advance(int(rng.uniform(14, 28)))
            turn = rng.choice([-1, 1]) * (math.pi / 2)
            n_turn = 12
            advance(n_turn, turn / n_turn)
            advance(int(rng.uniform(14, 28)))
        elif style == "U_turn":
            advance(int(rng.uniform(10, 20)))
            turn = rng.choice([-1, 1]) * math.pi
            n_turn = 18
            advance(n_turn, turn / n_turn)
            advance(int(rng.uniform(10, 20)))
        elif style == "S_curve":
            advance(int(rng.uniform(8, 16)))
            s = rng.choice([-1, 1])
            advance(14, s * (math.pi / 2) / 14)
            advance(14, -s * (math.pi / 2) / 14)
            advance(int(rng.uniform(8, 16)))
        elif style == "arc":
            total = rng.uniform(math.pi * 0.6, math.pi * 1.2)
            n = int(rng.uniform(30, 55))
            advance(n, (rng.choice([-1, 1]) * total) / n)
        else:  # "niches" -> straightish corridor (niches added as extra walls later)
            advance(int(rng.uniform(30, 55)))
        return pts

    def _normals(self, cl) -> List[Vec]:
        normals = []
        for i in range(len(cl)):
            a = cl[max(0, i - 1)]
            b = cl[min(len(cl) - 1, i + 1)]
            tx, ty = b[0] - a[0], b[1] - a[1]
            n = math.hypot(tx, ty) or 1.0
            normals.append((-ty / n, tx / n))   # left normal
        return normals

    def _build_walls_from_centerline(self, cl, rng) -> None:
        hw = self.half_width
        normals = self._normals(cl)
        left = [(cl[i][0] + normals[i][0] * hw, cl[i][1] + normals[i][1] * hw)
                for i in range(len(cl))]
        right = [(cl[i][0] - normals[i][0] * hw, cl[i][1] - normals[i][1] * hw)
                 for i in range(len(cl))]
        segs: List[Seg] = []
        for i in range(len(cl) - 1):
            segs.append((left[i][0], left[i][1], left[i + 1][0], left[i + 1][1]))
            segs.append((right[i][0], right[i][1], right[i + 1][0], right[i + 1][1]))
        # end caps (close the tube at both ends, leaving the very ends open-ish)
        segs.append((left[0][0], left[0][1], right[0][0], right[0][1]))
        segs.append((left[-1][0], left[-1][1], right[-1][0], right[-1][1]))

        # optional dead-end niches that jut INTO the corridor (style "niches")
        if self.style == "niches":
            k = int(rng.uniform(2, 4))
            idxs = rng.choice(range(4, len(cl) - 4), size=min(k, max(1, len(cl) - 8)),
                              replace=False)
            for idx in np.atleast_1d(idxs):
                side = rng.choice([-1, 1])
                depth = hw * float(rng.uniform(0.5, 0.85))
                nx, ny = normals[idx]
                cx, cy = cl[idx]
                # a small box poking in from one wall -> robot must go around it
                bx, by = cx + side * nx * (hw - depth), cy + side * ny * (hw - depth)
                w2 = 0.35
                tx, ty = (cl[idx + 1][0] - cx), (cl[idx + 1][1] - cy)
                tn = math.hypot(tx, ty) or 1.0
                ux, uy = tx / tn, ty / tn
                p1 = (bx - ux * w2, by - uy * w2)
                p2 = (bx + ux * w2, by + uy * w2)
                tip = (bx + side * nx * depth, by + side * ny * depth)
                segs.append((p1[0], p1[1], tip[0], tip[1]))
                segs.append((p2[0], p2[1], tip[0], tip[1]))
                segs.append((p1[0], p1[1], p2[0], p2[1]))
        self.segments = segs

    # ------------------------------------------------------------------ #
    #  Geometry queries
    # ------------------------------------------------------------------ #
    def raycast(self, origin: Vec, angle: float, max_range: float) -> float:
        """Distance to the nearest wall along `angle`, clipped to max_range."""
        ox, oy = origin
        dx, dy = math.cos(angle), math.sin(angle)
        best = max_range
        for (x1, y1, x2, y2) in self.segments:
            # ray (o + t*d, t>=0) vs segment (a + u*(b-a), u in [0,1])
            ex, ey = x2 - x1, y2 - y1
            denom = dx * ey - dy * ex
            if abs(denom) < 1e-12:
                continue
            diffx, diffy = x1 - ox, y1 - oy
            t = (diffx * ey - diffy * ex) / denom
            u = (diffx * dy - diffy * dx) / denom
            if 0.0 <= t < best and 0.0 <= u <= 1.0:
                best = t
        return best

    def segment_hits_circle(self, p: Vec, r: float) -> bool:
        px, py = p
        r2 = r * r
        for (x1, y1, x2, y2) in self.segments:
            # distance^2 from point to segment
            ex, ey = x2 - x1, y2 - y1
            L2 = ex * ex + ey * ey or 1e-12
            t = max(0.0, min(1.0, ((px - x1) * ex + (py - y1) * ey) / L2))
            cx, cy = x1 + t * ex, y1 + t * ey
            if (px - cx) ** 2 + (py - cy) ** 2 <= r2:
                return True
        return False

    # ------------------------------------------------------------------ #
    #  Vectorised batch raycast (numpy) -- one origin, many angles vs all
    #  segments at once. Much faster than calling raycast() in a Python loop;
    #  used by the SLAM scanner which casts a full fan every step.
    # ------------------------------------------------------------------ #
    def _seg_array(self):
        arr = getattr(self, "_seg_cache", None)
        if arr is None or arr.shape[0] != len(self.segments):
            arr = np.array(self.segments, dtype=np.float64) if self.segments \
                else np.zeros((0, 4), dtype=np.float64)
            self._seg_cache = arr
        return arr

    def raycast_batch(self, origin: Vec, angles, max_range: float):
        segs = self._seg_array()
        n = len(angles)
        if segs.shape[0] == 0:
            return np.full(n, max_range, dtype=np.float32)
        ox, oy = origin
        dx = np.cos(angles); dy = np.sin(angles)               # (n,)
        x1 = segs[:, 0]; y1 = segs[:, 1]                        # (m,)
        ex = segs[:, 2] - segs[:, 0]; ey = segs[:, 3] - segs[:, 1]
        # solve per (ray i, segment j): denom = dx_i*ey_j - dy_i*ex_j
        denom = dx[:, None] * ey[None, :] - dy[:, None] * ex[None, :]   # (n,m)
        diffx = (x1[None, :] - ox); diffy = (y1[None, :] - oy)          # (1,m)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (diffx * ey[None, :] - diffy * ex[None, :]) / denom
            u = (diffx * dy[:, None] - diffy * dx[:, None]) / denom
        valid = (np.abs(denom) > 1e-12) & (t >= 0.0) & (t < max_range) & (u >= 0.0) & (u <= 1.0)
        t = np.where(valid, t, np.inf)
        best = np.min(t, axis=1)
        best = np.where(np.isfinite(best), best, max_range)
        return np.clip(best, 0.0, max_range).astype(np.float32)
