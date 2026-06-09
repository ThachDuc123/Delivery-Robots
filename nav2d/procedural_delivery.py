"""Procedural delivery worlds for generalization (Level-2: train on infinite maps).

Each call builds a NEW random multi-corridor map (reusing the occupancy-grid
generators from ``world_hard.py``), then auto-places a **dock** and **N delivery
points** on free cells that are spread out and all reachable from the dock.

This is what lets the policy generalise: every episode is a different building,
so the network must learn the *rule* ("follow the planned route, avoid walls by
LiDAR") rather than memorising specific maps. A held-out set of hand-made fixed
maps (apartment_a/b/c) is then used to measure true zero-shot transfer.

Returns the same dict shape as ``fixed_maps.build_map``:
    {world, dock, points{ id:(x,y) }, grid, cell, origin, name}
so the planner / env / renderer all work unchanged.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from world2d import World
from world_hard import _make_grid, _boundary_segments, HARD_STYLES


def _free_cells(grid):
    return list(zip(*np.where(grid == 1)))


def _bfs_reachable(grid, start):
    R, C = grid.shape
    seen = {start}; q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in ((r+1, c), (r-1, c), (r, c+1), (r, c-1)):
            if 0 <= nr < R and 0 <= nc < C and (nr, nc) not in seen and grid[nr, nc] == 1:
                seen.add((nr, nc)); q.append((nr, nc))
    return seen


def _interior_score(grid, r, c):
    """How enclosed a free cell is (more walls around -> niche-like). Used to
    bias some delivery points into niches/dead-ends, like real lockers."""
    R, C = grid.shape
    walls = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < R and 0 <= cc < C and grid[rr, cc] == 1):
                walls += 1
    return walls


# --------------------------------------------------------------------------- #
#  Curved / dead-end carving (the missing structures: arcs, S-bends, U dead-ends,
#  nested niches) so the policy actually trains on point-5-like geometry.
# --------------------------------------------------------------------------- #
def _carve_disk(grid, r, c, half):
    R, C = grid.shape
    for dr in range(-half, half + 1):
        for dc in range(-half, half + 1):
            if dr * dr + dc * dc <= half * half + 1:
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < R and 0 <= cc < C:
                    grid[rr, cc] = 1


def _carve_polyline(grid, pts, half=1):
    """Carve a corridor of half-width `half` along a dense list of (r,c) points."""
    for (r, c) in pts:
        _carve_disk(grid, r, c, half)


def _arc_pts(cr, cc, rad, a0, a1, steps=80):
    return [(cr + rad * math.sin(a0 + (a1 - a0) * k / steps),
             cc + rad * math.cos(a0 + (a1 - a0) * k / steps)) for k in range(steps + 1)]


def _curved_grid(rng):
    """Build a grid whose MAIN corridor is curved (arc / S / zigzag) and which has
    dead-end niches (incl. a nested one) hanging off it. Returns the grid."""
    R = int(rng.integers(28, 38)); C = int(rng.integers(28, 40))
    g = np.zeros((R, C), dtype=np.uint8)
    half = 1
    kind = rng.choice(["arc", "s_curve", "zigzag", "u_turn"])
    cr, cc = R // 2, C // 2

    if kind == "arc":
        rad = rng.integers(min(R, C) // 3, min(R, C) // 2)
        a0 = math.radians(rng.uniform(-30, 30))
        a1 = a0 + math.radians(rng.uniform(120, 210))
        pts = _arc_pts(cr + rad // 2, cc, rad, a0, a1)
    elif kind == "s_curve":
        rad = rng.integers(5, 9)
        p1 = _arc_pts(cr - rad, 6 + rad, rad, math.radians(180), math.radians(0), 50)
        p2 = _arc_pts(cr + rad, 6 + rad + 2 * rad, rad, math.radians(180), math.radians(360), 50)
        pts = p1 + p2
    elif kind == "zigzag":
        pts = []; r, c = 4, 4; pts.append((r, c)); n = int(rng.integers(3, 5))
        for i in range(n):
            c2 = min(C - 4, c + int(rng.integers(5, 9)))
            pts += [(r, x) for x in range(c, c2)]
            r2 = r + int(rng.choice([-1, 1])) * int(rng.integers(5, 8))
            r2 = max(4, min(R - 4, r2))
            pts += [(y, c2) for y in range(min(r, r2), max(r, r2))]
            r, c = r2, c2
    else:  # u_turn: go right, hairpin, come back left
        rad = rng.integers(4, 7); ytop = cr - rad; ybot = cr + rad
        pts = [(ytop, x) for x in range(5, C - 6)]
        pts += _arc_pts(cr, C - 6, rad, math.radians(-90), math.radians(90), 40)
        pts += [(ybot, x) for x in range(C - 6, 5, -1)]
    _carve_polyline(g, pts, half)

    # dead-end niches off the corridor (one nested) -> point-5-like targets
    free = list(zip(*np.where(g == 1)))
    n_niche = int(rng.integers(2, 5))
    for _ in range(n_niche):
        br, bc = free[rng.integers(len(free))]
        d = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        depth = int(rng.integers(3, 7))
        line = [(br + d[0] * k, bc + d[1] * k) for k in range(depth)]
        _carve_polyline(g, line, half)
        # nested sub-niche off the end of some niches
        if rng.random() < 0.5:
            er, ec = line[-1]; d2 = (d[1], d[0])  # perpendicular
            _carve_polyline(g, [(er + d2[0] * k, ec + d2[1] * k) for k in range(int(rng.integers(2, 5)))], 0)
    return g


def build_procedural(rng: np.random.Generator, n_points: Optional[int] = None,
                     cell_range=(0.55, 0.75), style: Optional[str] = None,
                     curved_ratio: float = 0.5) -> Dict:
    """Generate a random delivery map with an auto-placed dock + delivery points.

    ``curved_ratio`` = fraction of episodes that use a CURVED map (arc/S/zigzag/U
    + dead-ends); the rest use grid-style maps. 50/50 by default = the mixed
    curriculum that keeps both straight-corridor and curve skills."""
    for _attempt in range(20):
        if style == "curved" or (style is None and rng.random() < curved_ratio):
            st = "curved"; grid = _curved_grid(rng)
        else:
            st = style or rng.choice(HARD_STYLES)
            grid, _nodes = _make_grid(rng, st)
        free = _free_cells(grid)
        if len(free) < 30:
            continue
        # dock = a well-connected free cell (many free neighbours)
        dock_cell = min(free, key=lambda f: _interior_score(grid, *f))
        reach = _bfs_reachable(grid, dock_cell)
        reach = [c for c in reach]
        if len(reach) < 25:
            continue

        cell = float(rng.uniform(*cell_range))
        R, C = grid.shape
        ox = -(C * cell) / 2.0; oy = -(R * cell) / 2.0
        def to_world(rc):
            r, c = rc
            return (ox + (c + 0.5) * cell, oy + (r + 0.5) * cell)

        # pick N points: spread out (min grid distance apart), some biased to niches
        k = n_points if n_points is not None else int(rng.integers(6, 11))
        candidates = sorted(reach, key=lambda c: -_interior_score(grid, *c))  # niche-first
        chosen: List[Tuple[int, int]] = []
        min_sep = max(4, (R + C) // 10)
        # always allow corridor points too: interleave niche-biased and random
        pool = candidates[: max(len(candidates) // 2, k * 3)] + list(rng.permutation(reach))
        for cand in pool:
            cand = tuple(int(v) for v in cand)
            if cand == dock_cell:
                continue
            if all(abs(cand[0]-p[0]) + abs(cand[1]-p[1]) >= min_sep for p in chosen):
                chosen.append(cand)
            if len(chosen) >= k:
                break
        if len(chosen) < max(3, k // 2):
            continue

        world = World(half_width=cell * 1.5, style=f"proc:{st}")
        world.segments = _boundary_segments(grid, cell, ox, oy)
        world.bounds = (ox - cell * 2, ox + C * cell + cell * 2,
                        oy - cell * 2, oy + R * cell + cell * 2)
        points = {i: to_world(c) for i, c in enumerate(chosen)}
        return {"world": world, "dock": to_world(dock_cell), "points": points,
                "grid": grid, "cell": cell, "origin": (ox, oy), "name": f"proc:{st}"}

    # fallback: a straight corridor (should basically never happen)
    from world2d import World as W2
    w = W2.generate(rng, style="straight")
    return {"world": w, "dock": w.start, "points": {0: w.goal}, "grid": None,
            "cell": 0.6, "origin": (0, 0), "name": "proc:fallback"}
