"""Hand-designed fixed maps for multi-stop delivery (nav2d).

Two apartment-style floors, each a wide main corridor plus many side niches —
including a **curved (arc) niche** and several **narrow/hard niches** — with
8-12 fixed **delivery points** (numbered locations) and a **dock** (charging /
start). The robot is told *which* points to deliver to; a TSP planner orders the
visits and routes home to the dock to save battery.

Built on the same occupancy-grid -> wall-segment representation as
``world_hard.py`` so it is API-compatible with ``world2d.World`` (segments,
raycast, segment_hits_circle, bounds) and works with the existing env / planner /
renderer unchanged.

Each map exposes:
  build_map(name) -> dict(world, dock (x,y), points {id:(x,y)}, grid, cell, origin)
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from world2d import World


# --------------------------------------------------------------------------- #
#  grid carving helpers (cell value 1 = free corridor)
# --------------------------------------------------------------------------- #
def _blank(R, C):
    return np.zeros((R, C), dtype=np.uint8)


def _h(g, r, c0, c1, half=1):
    for c in range(min(c0, c1), max(c0, c1) + 1):
        for dr in range(-half, half + 1):
            if 0 <= r + dr < g.shape[0] and 0 <= c < g.shape[1]:
                g[r + dr, c] = 1


def _v(g, c, r0, r1, half=1):
    for r in range(min(r0, r1), max(r0, r1) + 1):
        for dc in range(-half, half + 1):
            if 0 <= r < g.shape[0] and 0 <= c + dc < g.shape[1]:
                g[r, c + dc] = 1


def _rect(g, r0, r1, c0, c1):
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if 0 <= r < g.shape[0] and 0 <= c < g.shape[1]:
                g[r, c] = 1


def _arc(g, cr, cc, rad, a0, a1, half=1, steps=60):
    """Carve a curved corridor (centre cr,cc) from angle a0->a1 (radians)."""
    for k in range(steps + 1):
        a = a0 + (a1 - a0) * k / steps
        r = int(round(cr + rad * math.sin(a)))
        c = int(round(cc + rad * math.cos(a)))
        for dr in range(-half, half + 1):
            for dc in range(-half, half + 1):
                if 0 <= r + dr < g.shape[0] and 0 <= c + dc < g.shape[1]:
                    g[r + dr, c + dc] = 1


def _boundary_segments(grid, cell, ox, oy):
    R, C = grid.shape
    segs = []
    def w(r, c): return not (0 <= r < R and 0 <= c < C and grid[r, c] == 1)
    for r in range(R):
        for c in range(C):
            if grid[r, c] != 1:
                continue
            x0 = ox + c * cell; y0 = oy + r * cell; x1 = x0 + cell; y1 = y0 + cell
            if w(r, c - 1): segs.append((x0, y0, x0, y1))
            if w(r, c + 1): segs.append((x1, y0, x1, y1))
            if w(r - 1, c): segs.append((x0, y0, x1, y0))
            if w(r + 1, c): segs.append((x0, y1, x1, y1))
    return segs


def _free_near(grid, r, c, rad=4):
    """Nearest free cell to (r,c) within `rad` (so points land inside corridors)."""
    if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c] == 1:
        return (r, c)
    best = None; bd = 1e9
    for dr in range(-rad, rad + 1):
        for dc in range(-rad, rad + 1):
            rr, cc = r + dr, c + dc
            if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1] and grid[rr, cc] == 1:
                d = dr * dr + dc * dc
                if d < bd:
                    bd = d; best = (rr, cc)
    return best


# --------------------------------------------------------------------------- #
#  Map definitions
# --------------------------------------------------------------------------- #
def _map_apartment_a():
    """Wide horizontal corridor + 6 vertical niches (one narrow), one arc niche,
    plus a small room. ~10 delivery points."""
    R, C, cell = 34, 46, 0.6
    g = _blank(R, C)
    midr = 16
    _h(g, midr, 3, C - 4, half=1)                 # main corridor (wide)

    # vertical niches up/down off the main corridor. All are half=1 (3 cells ~1.8m)
    # so a round robot can enter, reach the end, turn around and come back out --
    # narrower than that (1 cell) is physically too tight for a 0.44 m robot to
    # U-turn in a dead end, so the "hard" niches stay tight-but-traversable.
    niche_cols = [7, 13, 20, 27, 33, 39]
    for i, c in enumerate(niche_cols):
        up = (i % 2 == 0)
        if up:
            _v(g, c, 4, midr, half=1)
        else:
            _v(g, c, midr, R - 5, half=1)

    # a small room at the right end (multi-cell)
    _rect(g, midr - 4, midr + 4, C - 9, C - 5)

    # an ARC niche curving down-left from the corridor near c=10
    _arc(g, midr, 10, 8, a0=math.radians(95), a1=math.radians(175), half=1)

    cell_pts = {
        0: (4, midr),            # corridor west end
        1: (4, 7),               # top niche end
        2: (R - 6, 13),          # bottom niche end
        3: (4, 20),              # top niche end
        4: (R - 6, 27),          # bottom niche end
        5: (4, 33),              # NARROW niche end (hard)
        6: (midr, C - 7),        # room centre
        7: (midr + 7, 2),        # arc niche far end (curved)
        8: (R - 6, 39),          # bottom niche end (far)
        9: (midr, C - 5),        # corridor east end
    }
    dock_cell = (midr, 4)
    return R, C, cell, g, cell_pts, dock_cell


def _map_apartment_b():
    """Two parallel corridors joined by cross-links (loop-ish) + niches + arc.
    More junctions => the TSP ordering matters more. ~12 delivery points."""
    R, C, cell = 40, 44, 0.6
    g = _blank(R, C)
    r_top, r_bot = 10, 30
    _h(g, r_top, 4, C - 5, half=1)
    _h(g, r_bot, 4, C - 5, half=1)
    for c in (6, 16, 26, 36):                     # vertical cross-links
        _v(g, c, r_top, r_bot, half=1)
    # niches off the top and bottom corridors
    for c in (11, 21, 31):
        _v(g, c, 3, r_top, half=1)                # up from top corridor
    for c in (11, 31):
        _v(g, c, r_bot, R - 4, half=(0 if c == 21 else 1))
    _v(g, 21, r_bot, R - 4, half=1)               # niche (tight-but-traversable)
    # arc niche curving off the bottom-right: one end sits ON the bottom corridor
    # (c=39, the corridor's east end) and curves down-left so it stays connected.
    _arc(g, r_bot, 33, 6, a0=math.radians(0), a1=math.radians(95), half=1)

    cell_pts = {
        0: (r_top, 5),
        1: (r_top, C - 6),
        2: (r_bot, 5),
        3: (r_bot, C - 6),
        4: (3, 11),
        5: (3, 21),
        6: (3, 31),
        7: (R - 5, 11),
        8: (R - 5, 21),          # narrow niche end (hard)
        9: (r_top, 16),
        10: (r_bot, 26),
        11: (R - 6, C - 6),      # arc niche end
    }
    dock_cell = (r_top, 6)
    return R, C, cell, g, cell_pts, dock_cell


def _map_apartment_c():
    """Harder map: wide main corridor + several vertical niches, plus a big
    semicircular ARC corridor whose MID has a sub-niche branching off it (a
    delivery point sits inside that arc sub-niche). Designed for ~3-stop trips."""
    R, C, cell = 40, 48, 0.6
    g = _blank(R, C)
    midr = 14
    _h(g, midr, 3, C - 4, half=1)                      # main corridor (top)

    # vertical niches off the main corridor (delivery points live at their ends)
    for c in (8, 15, 22, 29, 36, 42):
        _v(g, c, 4, midr, half=1)                       # niches going UP

    # big arc: semicircle hanging BELOW the corridor, centre (midr, arc_cc)
    arc_cc = 16; r_in = 7; r_out = r_in + 2
    # two concentric arc walls + carve the ring between them as a corridor
    for rad in range(r_in, r_out + 1):
        _arc(g, midr, arc_cc, rad, a0=math.radians(0), a1=math.radians(180), half=0)
    # connect both arc mouths to the main corridor
    _v(g, arc_cc - (r_in + 1), midr, midr + 1, half=0)
    _v(g, arc_cc + (r_in + 1), midr, midr + 1, half=0)
    # SUB-NICHE off the MIDDLE of the arc (bottom of the semicircle) going down
    arc_bottom_r = midr + (r_in + r_out) // 2
    _v(g, arc_cc, arc_bottom_r, arc_bottom_r + 6, half=1)

    cell_pts = {
        0: (4, 8),                  # niche end (up)
        1: (4, 22),                 # niche end (up)
        2: (4, 36),                 # niche end (up)
        3: (4, 42),                 # niche end (up, far)
        4: (midr, C - 5),           # corridor east end
        5: (arc_bottom_r + 5, arc_cc),   # INSIDE the arc sub-niche (hard, curved approach)
        6: (midr + r_in + 1, arc_cc - r_in - 1),  # arc left arm
        7: (midr + r_in + 1, arc_cc + r_in + 1),  # arc right arm
    }
    dock_cell = (midr, 5)
    return R, C, cell, g, cell_pts, dock_cell


def _polyline(g, pts, half=1):
    """Carve a corridor along a dense list of (r,c) float points."""
    R, C = g.shape
    for (rf, cf) in pts:
        r, c = int(round(rf)), int(round(cf))
        for dr in range(-half, half + 1):
            for dc in range(-half, half + 1):
                if 0 <= r + dr < R and 0 <= c + dc < C:
                    g[r + dr, c + dc] = 1


def _map_test_c_curve():
    """Zero-shot curve test: a long half-circle (C) corridor. Tests holding a
    steady steering angle for a long time to hug a curved wall."""
    R, C, cell = 40, 40, 0.6
    g = _blank(R, C)
    cr, cc, rad = 10, 20, 13
    arc = [(cr + rad * math.sin(a), cc + rad * math.cos(a))
           for a in np.linspace(0, math.pi, 120)]   # right tip -> top -> left tip
    _polyline(g, arc, half=1)
    cell_pts = {0: (int(arc[10][0]), int(arc[10][1])),   # near right end of the arc
                1: (cr + rad, cc),                       # top of the arc
                2: (int(arc[-10][0]), int(arc[-10][1]))}  # near left end
    dock_cell = (int(arc[0][0]), int(arc[0][1]))         # right tip
    return R, C, cell, g, cell_pts, dock_cell


def _map_test_s_curve():
    """Zero-shot curve test: a smooth sine S-curve -> tests switching steering
    direction left<->right continuously."""
    R, C, cell = 32, 50, 0.6
    g = _blank(R, C)
    midr, amp = 16, 9
    s = [(midr + amp * math.sin(2 * math.pi * (x - 5) / 28.0), x)
         for x in np.linspace(5, C - 5, 220)]
    _polyline(g, s, half=1)
    cell_pts = {0: (int(s[20][0]), int(s[20][1])),
                1: (int(s[len(s)//2][0]), int(s[len(s)//2][1])),
                2: (int(s[-15][0]), int(s[-15][1]))}
    dock_cell = (int(s[0][0]), int(s[0][1]))
    return R, C, cell, g, cell_pts, dock_cell


def _map_test_u_turn():
    """Zero-shot curve test: a tight U / hairpin at the end of a hall -> tests
    slowing & hugging a sharp 180-degree bend."""
    R, C, cell = 36, 40, 0.6
    g = _blank(R, C)
    rtop, rbot = 12, 22; rad = (rbot - rtop) // 2
    _h(g, rtop, 4, C - 8, half=1)
    _h(g, rbot, 4, C - 8, half=1)
    _arc(g, (rtop + rbot) // 2, C - 8, rad, math.radians(-90), math.radians(90), half=1)  # hairpin
    cell_pts = {0: (rtop, 8), 1: (rbot, 10), 2: ((rtop + rbot) // 2, C - 7)}
    dock_cell = (rtop, 6)
    return R, C, cell, g, cell_pts, dock_cell


_MAPS = {"apartment_a": _map_apartment_a, "apartment_b": _map_apartment_b,
         "apartment_c": _map_apartment_c,
         "test_c_curve": _map_test_c_curve, "test_s_curve": _map_test_s_curve,
         "test_u_turn": _map_test_u_turn}


def map_names() -> List[str]:
    return list(_MAPS)


def build_map(name: str = "apartment_a") -> Dict:
    R, C, cell, g, cell_pts, dock_cell = _MAPS[name]()
    ox = -(C * cell) / 2.0
    oy = -(R * cell) / 2.0

    def to_world(rc):
        r, c = rc
        return (ox + (c + 0.5) * cell, oy + (r + 0.5) * cell)

    # snap every point/dock to the nearest free cell so none sit in a wall
    points = {}
    for pid, rc in cell_pts.items():
        fc = _free_near(g, rc[0], rc[1], rad=5)
        points[pid] = to_world(fc if fc else rc)
    dfc = _free_near(g, dock_cell[0], dock_cell[1], rad=5) or dock_cell
    dock = to_world(dfc)

    world = World(half_width=cell * 1.5, style=f"fixed:{name}")
    world.segments = _boundary_segments(g, cell, ox, oy)
    world.bounds = (ox - cell * 2, ox + C * cell + cell * 2,
                    oy - cell * 2, oy + R * cell + cell * 2)
    # centerline left empty (not used for fixed maps); start/goal set per task
    return {"world": world, "dock": dock, "points": points,
            "grid": g, "cell": cell, "origin": (ox, oy), "name": name}
