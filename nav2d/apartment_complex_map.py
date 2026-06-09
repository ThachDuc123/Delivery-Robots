"""apartment_complex_v1 — a brand-new apartment-floor map (coordinate spec).

Distinct from every earlier map (warehouse / apartment_a-c / test_*). Theme: one
floor of a large apartment complex.

Layout (grid coords, 1 cell = 0.40 m; r = row/y, c = col/x):
  * a WIDE main corridor in a **T shape**: a long horizontal spine + a vertical
    branch going up (the floor's two circulation axes),
  * apartment-door NICHES branching perpendicular off both corridors,
  * a CENTRAL LIGHT-WELL (giếng trời): a solid square block with a **curved ring
    corridor wrapping around it** — built from TWO connected arcs (a full loop),
  * an ISOLATED DOCK (technical/refuse room) walled off with a SINGLE narrow
    door, placed away from the resident walking routes,
  * 4 DESTINATIONS (apartment doors) deep in straight niches + on the arc ring,
    all far from the dock,
  * RESIDENT patrol routes (waypoint loops) along the corridors.

`build()` rasterises to an occupancy grid; `export_config()` returns the detailed
coordinate dict and writes JSON.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np

CELL = 0.40
ROWS, COLS = 54, 64                         # ~21.6 x 25.6 m floor
ORIGIN = (-COLS * CELL / 2.0, -ROWS * CELL / 2.0)

# --- T-shaped main corridor (5 cells wide ~2.0 m) -------------------------- #
H_CORR = dict(r0=30, r1=34, c0=5, c1=58)     # horizontal spine
V_CORR = dict(c0=28, c1=32, r0=8, r1=34)     # vertical branch up to the spine

# --- apartment-door straight niches: (col_or_row, side, depth) ------------- #
# off the horizontal spine (up / down). More niches -> more apartment doors.
# Cols are kept clear of the vertical branch (28-32) and the ring area (38-50).
H_NICHES = [(10, "up", 7), (16, "down", 6), (22, "up", 6),
            (52, "up", 7), (56, "down", 6)]
# off the vertical branch (left / right)
V_NICHES = [(14, "left", 6), (22, "right", 6)]

# --- central light-well + wrap-around curved ring -------------------------- #
WELL = dict(cr=18, cc=44, half=3)            # solid block (giếng trời) center+halfsize
RING = dict(cr=18, cc=44, rad=6)             # curved corridor radius around the well
RING_LINK = dict(r0=24, r1=30, c=44)         # short link from ring down to the spine

# --- isolated dock (technical room) bottom-left, one narrow door ----------- #
DOCK_ROOM = dict(r0=44, r1=51, c0=6, c1=14)
DOCK_DOOR = dict(r=43, c=10)                 # single 1-cell door
DOCK_STUB = dict(r0=27, r1=43, c=10)         # narrow vertical leg (1 cell wide)
DOCK_ELBOW_C = 18                            # then a narrow horizontal leg over to col 18 -> up into spine
DOCK_CELL = (48, 10)

# destinations are filled in _carve() from the actual niche/ring geometry so they
# always land at the deep end of a real corridor (not floating in a wall).
DEST_CELLS = {0: None, 1: None, 2: None, 3: None}

# resident patrol loops (cell waypoints) along corridors
PED_ROUTES_CELLS = [
    [(32, 8), (32, 26), (32, 8)],            # left half of the spine
    [(32, 56), (32, 36), (32, 56)],          # right half of the spine
    [(10, 30), (28, 30), (10, 30)],          # up/down the vertical branch
]


def _blank():
    return np.zeros((ROWS, COLS), dtype=np.uint8)


def _rect(g, r0, r1, c0, c1, val=1):
    g[max(0, r0):min(ROWS, r1 + 1), max(0, c0):min(COLS, c1 + 1)] = val


def _vstrip(g, c, r0, r1, half=1):
    for r in range(min(r0, r1), max(r0, r1) + 1):
        for dc in range(-half, half + 1):
            if 0 <= r < ROWS and 0 <= c + dc < COLS:
                g[r, c + dc] = 1


def _hstrip(g, r, c0, c1, half=1):
    for c in range(min(c0, c1), max(c0, c1) + 1):
        for dr in range(-half, half + 1):
            if 0 <= r + dr < ROWS and 0 <= c < COLS:
                g[r + dr, c] = 1


def _arc(g, cr, cc, rad, a0, a1, half=1, steps=120):
    pts = []
    for k in range(steps + 1):
        a = a0 + (a1 - a0) * k / steps
        r = cr + rad * math.sin(a); c = cc + rad * math.cos(a)
        ri, ci = int(round(r)), int(round(c)); pts.append((ri, ci))
        for dr in range(-half, half + 1):
            for dc in range(-half, half + 1):
                if 0 <= ri + dr < ROWS and 0 <= ci + dc < COLS:
                    g[ri + dr, ci + dc] = 1
    return pts


def _cell_to_world(rc):
    r, c = rc; ox, oy = ORIGIN
    return (ox + (c + 0.5) * CELL, oy + (r + 0.5) * CELL)


def _nearest_free(g, rc, rad=5):
    if 0 <= rc[0] < ROWS and 0 <= rc[1] < COLS and g[rc[0], rc[1]] == 1:
        return rc
    for k in range(1, rad + 1):
        for dr in range(-k, k + 1):
            for dc in range(-k, k + 1):
                r, c = rc[0] + dr, rc[1] + dc
                if 0 <= r < ROWS and 0 <= c < COLS and g[r, c] == 1:
                    return (r, c)
    return rc


def _carve(g):
    _rect(g, H_CORR["r0"], H_CORR["r1"], H_CORR["c0"], H_CORR["c1"])      # horizontal spine
    _rect(g, V_CORR["r0"], V_CORR["r1"], V_CORR["c0"], V_CORR["c1"])      # vertical branch
    for (c, d, dp) in H_NICHES:                                           # spine niches
        if d == "up": _vstrip(g, c, H_CORR["r0"] - dp, H_CORR["r0"])
        else:         _vstrip(g, c, H_CORR["r1"], H_CORR["r1"] + dp)
    for (r, d, dp) in V_NICHES:                                           # branch niches
        if d == "left":  _hstrip(g, r, V_CORR["c0"] - dp, V_CORR["c0"])
        else:            _hstrip(g, r, V_CORR["c1"], V_CORR["c1"] + dp)
    # curved ring around the light-well: two connected arcs forming a loop
    cr, cc, rad = RING["cr"], RING["cc"], RING["rad"]
    _arc(g, cr, cc, rad, math.radians(0), math.radians(180))             # top half-ring
    a2 = _arc(g, cr, cc, rad, math.radians(180), math.radians(360))      # bottom half-ring
    _vstrip(g, RING_LINK["c"], RING_LINK["r0"], RING_LINK["r1"])         # link ring -> spine
    # re-stamp the solid light-well block (so the ring's inner edge is a wall)
    _rect(g, WELL["cr"] - WELL["half"], WELL["cr"] + WELL["half"],
          WELL["cc"] - WELL["half"], WELL["cc"] + WELL["half"], val=0)
    # isolated dock room + single door + L-shaped exit passage. The passage is
    # now 3 cells wide (~1.2 m) so the robot drives out comfortably, while the
    # dock stays isolated (one passage only).
    _rect(g, DOCK_ROOM["r0"], DOCK_ROOM["r1"], DOCK_ROOM["c0"], DOCK_ROOM["c1"])
    _vstrip(g, DOCK_DOOR["c"], DOCK_DOOR["r"], DOCK_DOOR["r"], half=2)     # door (5 cells wide)
    _vstrip(g, DOCK_STUB["c"], DOCK_STUB["r0"], DOCK_STUB["r1"], half=2)   # vertical leg (5 cells)
    _hstrip(g, DOCK_STUB["r0"], DOCK_STUB["c"], DOCK_ELBOW_C, half=2)      # horizontal leg (5 cells)
    _vstrip(g, DOCK_ELBOW_C, DOCK_STUB["r0"], H_CORR["r1"], half=2)        # up into the spine (5 cells)
    # One destination (apartment door) at the DEEP END of EVERY niche, plus a
    # couple on the curved ring -> 8 delivery points total.
    dests = {}
    k = 0
    for (c, d, dp) in H_NICHES:
        rc = (H_CORR["r0"] - dp + 1, c) if d == "up" else (H_CORR["r1"] + dp - 1, c)
        dests[k] = rc; k += 1
    for (r, d, dp) in V_NICHES:
        rc = (r, V_CORR["c0"] - dp + 1) if d == "left" else (r, V_CORR["c1"] + dp - 1)
        dests[k] = rc; k += 1
    # one door on the curved ring (bottom of the wrap-around) -> 8 doors total
    dests[k] = (int(round(a2[len(a2) // 2][0])), int(round(a2[len(a2) // 2][1]))); k += 1
    return {"dests": dests}


def _grid_to_segments(g):
    segs = []
    def wall(r, c): return not (0 <= r < ROWS and 0 <= c < COLS and g[r, c] == 1)
    ox, oy = ORIGIN
    for r in range(ROWS):
        for c in range(COLS):
            if g[r, c] != 1: continue
            x0 = ox + c * CELL; y0 = oy + r * CELL; x1 = x0 + CELL; y1 = y0 + CELL
            if wall(r, c - 1): segs.append((x0, y0, x0, y1))
            if wall(r, c + 1): segs.append((x1, y0, x1, y1))
            if wall(r - 1, c): segs.append((x0, y0, x1, y0))
            if wall(r + 1, c): segs.append((x0, y1, x1, y1))
    return segs


def build() -> Dict:
    from world2d import World
    g = _blank(); meta = _carve(g)
    dests = {k: _nearest_free(g, v) for k, v in meta["dests"].items()}
    dock_cell = _nearest_free(g, DOCK_CELL)
    world = World(half_width=CELL * 2.5, style="apartment_complex_v1")
    world.segments = _grid_to_segments(g)
    xmin, ymin = ORIGIN
    world.bounds = (xmin - CELL, xmin + COLS * CELL + CELL,
                    ymin - CELL, ymin + ROWS * CELL + CELL)
    points = {k: _cell_to_world(v) for k, v in dests.items()}
    dock = _cell_to_world(dock_cell)
    ped_routes = [[_cell_to_world(_nearest_free(g, wp)) for wp in route]
                  for route in PED_ROUTES_CELLS]
    return {"world": world, "dock": dock, "points": points, "grid": g, "cell": CELL,
            "origin": ORIGIN, "name": "apartment_complex_v1", "ped_routes": ped_routes,
            "dest_cells": dests, "dock_cell": dock_cell}


def export_config(path: str = None) -> dict:
    m = build()
    cfg = {
        "name": "apartment_complex_v1",
        "map_bounds_world": {"xmin": m["world"].bounds[0], "xmax": m["world"].bounds[1],
                             "ymin": m["world"].bounds[2], "ymax": m["world"].bounds[3]},
        "grid": {"rows": ROWS, "cols": COLS, "cell_size_m": CELL,
                 "origin_world_xy": list(ORIGIN),
                 "cell_center_formula": "x=origin_x+(c+0.5)*cell ; y=origin_y+(r+0.5)*cell"},
        "corridors_cells": {
            "horizontal_spine": H_CORR, "vertical_branch": V_CORR,
            "width_m": (H_CORR["r1"] - H_CORR["r0"] + 1) * CELL,
            "apartment_niches_off_spine": [{"col": c, "dir": d, "depth_cells": dp} for c, d, dp in H_NICHES],
            "apartment_niches_off_branch": [{"row": r, "dir": d, "depth_cells": dp} for r, d, dp in V_NICHES],
            "curved_ring_around_lightwell": {"center_cell": [RING["cr"], RING["cc"]],
                                             "radius_cells": RING["rad"],
                                             "lightwell_block": WELL,
                                             "note": "two connected arcs (0-180 and 180-360) form a loop wrapping the well"},
        },
        "dock_isolated": {"room_cells": DOCK_ROOM, "single_door_cell": [DOCK_DOOR["r"], DOCK_DOOR["c"]],
                          "narrow_stub_cells": DOCK_STUB,
                          "dock_cell": list(m["dock_cell"]), "dock_world_xy": list(m["dock"]),
                          "isolation": "walled technical room; ONE 1-cell door + narrow stub, off the resident routes"},
        "destinations": {str(k): {"cell": list(m["dest_cells"][k]), "world_xy": list(m["points"][k]),
                                  "kind": ("ring/arc apartment" if k == 3 else "niche apartment door")}
                         for k in m["points"]},
        "resident_patrols": [
            {"waypoints_cells": [list(wp) for wp in route],
             "waypoints_world": [list(_cell_to_world(_nearest_free(m["grid"], wp))) for wp in route]}
            for route in PED_ROUTES_CELLS],
    }
    if path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    return cfg


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "apartment_complex_v1.json")
    export_config(out); print("wrote", out)
