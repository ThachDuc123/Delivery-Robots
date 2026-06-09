"""Hard multi-corridor 2D worlds (junctions, grids, loops) for nav2d.

Unlike `world2d.py` (a single corridor from one centerline), these maps connect
**many corridors**: T-junctions, 4-way crossings, full grids, branching trees and
loops. They are built on an **occupancy grid**:

  1. carve corridor cells (value 1 = free) into a grid of walls (0),
  2. extract wall **boundary segments** between free and wall cells -> the line
     segments the LiDAR raycasts against (same format as world2d.World),
  3. pick start + goal as two far-apart free cells that are connected (BFS),
  4. keep the BFS path as a "reference" (for spawn heading + drawing only; the
     policy never sees it).

Output object is API-compatible with world2d.World (segments, centerline,
start, goal, start_heading, bounds, raycast, segment_hits_circle) so the existing
env / render / eval code works unchanged.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from world2d import World  # reuse raycast + collision + dataclass shape

HARD_STYLES = ("T_junction", "cross", "grid", "branch", "loop", "double_T")


def _carve_line(grid, r0, c0, r1, c1, half):
    """Carve a straight (axis-aligned) corridor of half-width `half` cells."""
    R, C = grid.shape
    if r0 == r1:
        for c in range(min(c0, c1), max(c0, c1) + 1):
            for dr in range(-half, half + 1):
                rr = r0 + dr
                if 0 <= rr < R and 0 <= c < C:
                    grid[rr, c] = 1
    else:
        for r in range(min(r0, r1), max(r0, r1) + 1):
            for dc in range(-half, half + 1):
                cc = c0 + dc
                if 0 <= r < R and 0 <= cc < C:
                    grid[r, cc] = 1


def _make_grid(rng: np.random.Generator, style: str):
    """Return (occupancy grid, list of (r,c) junction/endpoint nodes)."""
    R = int(rng.integers(22, 30))
    C = int(rng.integers(22, 30))
    g = np.zeros((R, C), dtype=np.uint8)
    half = 1  # corridor half-width in cells (-> 3 cells wide)
    nodes = []

    def H(r, c0, c1): _carve_line(g, r, c0, r, c1, half);
    def V(c, r0, r1): _carve_line(g, r0, c, r1, c, half)

    midr, midc = R // 2, C // 2
    if style == "T_junction":
        H(midr, 3, C - 4); V(midc, 3, midr)
        nodes = [(midr, 3), (midr, C - 4), (3, midc)]
    elif style == "cross":
        H(midr, 3, C - 4); V(midc, 3, R - 4)
        nodes = [(midr, 3), (midr, C - 4), (3, midc), (R - 4, midc)]
    elif style == "double_T":
        H(midr, 3, C - 4)
        c1, c2 = C // 3, 2 * C // 3
        V(c1, 3, midr); V(c2, midr, R - 4)
        nodes = [(midr, 3), (midr, C - 4), (3, c1), (R - 4, c2)]
    elif style == "grid":
        rows = [R // 4, R // 2, 3 * R // 4]
        cols = [C // 4, C // 2, 3 * C // 4]
        for r in rows: H(r, cols[0], cols[-1])
        for c in cols: V(c, rows[0], rows[-1])
        nodes = [(rows[0], cols[0]), (rows[0], cols[-1]),
                 (rows[-1], cols[0]), (rows[-1], cols[-1]),
                 (rows[1], cols[1])]
    elif style == "branch":
        H(midr, 3, C - 4)
        for _ in range(int(rng.integers(2, 4))):
            bc = int(rng.integers(5, C - 5))
            up = rng.random() < 0.5
            V(bc, (3 if up else midr), (midr if up else R - 4))
            nodes.append(((3 if up else R - 4), bc))
        nodes += [(midr, 3), (midr, C - 4)]
    else:  # "loop" -> rectangular ring + one spur
        r0, r1 = R // 5, 4 * R // 5
        c0, c1 = C // 5, 4 * C // 5
        H(r0, c0, c1); H(r1, c0, c1); V(c0, r0, r1); V(c1, r0, r1)
        sc = (c0 + c1) // 2
        V(sc, r1, R - 4)
        nodes = [(r0, c0), (r0, c1), (r1, c0), (r1, c1), (R - 4, sc)]
    return g, nodes


def _boundary_segments(grid, cell: float, ox: float, oy: float) -> List[Tuple[float, float, float, float]]:
    """Edges between a free cell and a wall/outside cell -> wall segments (world)."""
    R, C = grid.shape
    segs = []

    def w(r, c):
        return not (0 <= r < R and 0 <= c < C and grid[r, c] == 1)  # True if wall/outside

    for r in range(R):
        for c in range(C):
            if grid[r, c] != 1:
                continue
            x0 = ox + c * cell; y0 = oy + r * cell
            x1 = x0 + cell; y1 = y0 + cell
            if w(r, c - 1): segs.append((x0, y0, x0, y1))   # left
            if w(r, c + 1): segs.append((x1, y0, x1, y1))   # right
            if w(r - 1, c): segs.append((x0, y0, x1, y0))   # bottom
            if w(r + 1, c): segs.append((x0, y1, x1, y1))   # top
    return segs


def _bfs_path(grid, s, g):
    R, C = grid.shape
    if grid[s] != 1 or grid[g] != 1:
        return None
    seen = {s}; par = {s: None}; q = deque([s])
    while q:
        cur = q.popleft()
        if cur == g:
            path = []
            while cur is not None:
                path.append(cur); cur = par[cur]
            return path[::-1]
        r, c = cur
        for nr, nc in ((r+1,c),(r-1,c),(r,c+1),(r,c-1)):
            if 0 <= nr < R and 0 <= nc < C and (nr,nc) not in seen and grid[nr,nc] == 1:
                seen.add((nr,nc)); par[(nr,nc)] = cur; q.append((nr,nc))
    return None


def generate_hard(rng: np.random.Generator, style: Optional[str] = None,
                  cell: float = 0.7) -> World:
    style = style or rng.choice(HARD_STYLES)
    for _attempt in range(8):
        grid, nodes = _make_grid(rng, style)
        free = list(zip(*np.where(grid == 1)))
        if len(free) < 20:
            continue
        # choose start/goal: two free cells, far apart, connected
        best = None
        cand = nodes if len(nodes) >= 2 else free
        for _ in range(40):
            s = tuple(int(v) for v in cand[rng.integers(len(cand))])
            g = tuple(int(v) for v in cand[rng.integers(len(cand))])
            if grid[s] != 1: s = min(free, key=lambda f: (f[0]-s[0])**2+(f[1]-s[1])**2)
            if grid[g] != 1: g = min(free, key=lambda f: (f[0]-g[0])**2+(f[1]-g[1])**2)
            d = abs(s[0]-g[0]) + abs(s[1]-g[1])
            if d < (grid.shape[0]+grid.shape[1])//2:
                continue
            path = _bfs_path(grid, s, g)
            if path and (best is None or len(path) > best[2]):
                best = (s, g, len(path), path)
        if best is None:
            continue
        s, g, _, path = best

        R, C = grid.shape
        ox = -(C * cell) / 2.0
        oy = -(R * cell) / 2.0
        segs = _boundary_segments(grid, cell, ox, oy)

        def cell_center(rc):
            r, c = rc
            return (ox + (c + 0.5) * cell, oy + (r + 0.5) * cell)

        w = World(half_width=cell * 1.5, style=style)
        w.segments = segs
        w.centerline = [cell_center(p) for p in path]
        w.start = cell_center(s)
        w.goal = cell_center(g)
        d0 = np.array(w.centerline[1]) - np.array(w.centerline[0]) if len(w.centerline) > 1 else np.array([1.0, 0.0])
        w.start_heading = math.atan2(d0[1], d0[0])
        xs = [p[0] for p in w.centerline]; ys = [p[1] for p in w.centerline]
        m = cell * 3
        w.bounds = (ox - m, ox + C * cell + m, oy - m, oy + R * cell + m)
        return w
    # fallback: a simple straight if generation kept failing
    from world2d import World as W2
    return W2.generate(rng, style="straight")
