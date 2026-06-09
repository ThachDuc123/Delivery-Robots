"""BƯỚC 1 — Domain Randomization: sinh 1000 bản đồ Occupancy Grid đa dạng.

Trộn 3 archetype để policy gặp đủ kiểu địa hình (chống overfit):
  * procedural / curved  (lưới hành lang + cung tròn / chữ S / chữ U)  -- reuse build_procedural
  * apartment            (ngách hẹp ~1.2m, cua chữ U, vòng cung giếng trời) -- reuse apartment_complex
  * hotel                (hành lang chính dài thẳng + nhiều nhánh vuông góc) -- generator riêng

Mỗi map lưu:  data/maps/map_XXXX.npy  = occupancy grid (uint8, 1=free, 0=wall)
Kèm:          data/maps/index.json    = {file, kind, cell, origin, dock, points}
sao cho BƯỚC 2 dựng lại được World + dock + điểm giao để train.

Chạy:  .venv\\Scripts\\python.exe generate_training_maps.py --n 1000
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

import procedural_delivery as pd
import apartment_complex_map as ac

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "maps")


# --------------------------- hotel archetype ------------------------------ #
def _carve_rect(g, r0, r1, c0, c1):
    R, C = g.shape
    g[max(0, r0):min(R, r1 + 1), max(0, c0):min(C, c1 + 1)] = 1


def build_hotel(rng) -> dict:
    """Long straight main corridor + perpendicular room branches (varied width)."""
    cell = 0.4
    R = rng.integers(40, 60); C = rng.integers(70, 100)
    g = np.zeros((R, C), dtype=np.uint8)
    # main spine (horizontal), width 4-6 cells (~1.6-2.4 m)
    sr = R // 2; hw = int(rng.integers(2, 4))
    c0, c1 = 4, C - 5
    _carve_rect(g, sr - hw, sr + hw, c0, c1)
    # sometimes a second parallel spine (double-loaded corridor)
    if rng.random() < 0.4:
        sr2 = sr + int(rng.integers(8, 14))
        if sr2 + hw < R - 2:
            _carve_rect(g, sr2 - hw, sr2 + hw, c0, c1)
            cc = int(rng.integers(c0 + 4, c1 - 4)); _carve_rect(g, sr, sr2, cc - 1, cc + 1)
    # perpendicular branches (rooms) off the spine, varied depth/width
    n_br = int(rng.integers(5, 10))
    pts = []
    for _ in range(n_br):
        bc = int(rng.integers(c0 + 3, c1 - 3))
        depth = int(rng.integers(5, 12)); bw = int(rng.integers(1, 3))  # 1-2 -> 3-5 cells
        up = rng.random() < 0.5
        if up:
            r_to = max(1, sr - hw - depth); _carve_rect(g, r_to, sr - hw, bc - bw, bc + bw)
            pts.append((r_to + 1, bc))
        else:
            r_to = min(R - 2, sr + hw + depth); _carve_rect(g, sr + hw, r_to, bc - bw, bc + bw)
            pts.append((r_to - 1, bc))
    # dock: an isolated end room off one tip via a short stub
    droom_c = c0 - 0 if False else c0
    _carve_rect(g, sr - hw, sr + hw, 1, c0)             # left stub to a dock pocket
    dock_cell = (sr, 2)
    return _finish(g, cell, "hotel", dock_cell, pts, rng)


# --------------------------- common finish -------------------------------- #
def _nearest_free(g, rc, rad=6):
    R, C = g.shape
    if 0 <= rc[0] < R and 0 <= rc[1] < C and g[rc[0], rc[1]] == 1:
        return rc
    for k in range(1, rad + 1):
        for dr in range(-k, k + 1):
            for dc in range(-k, k + 1):
                r, c = rc[0] + dr, rc[1] + dc
                if 0 <= r < R and 0 <= c < C and g[r, c] == 1:
                    return (r, c)
    return None


def _finish(g, cell, kind, dock_cell, point_cells, rng):
    R, C = g.shape
    ox = -C * cell / 2.0; oy = -R * cell / 2.0
    def to_world(rc): return (ox + (rc[1] + 0.5) * cell, oy + (rc[0] + 0.5) * cell)
    dock_cell = _nearest_free(g, dock_cell) or (R // 2, C // 2)
    pts = {}
    for i, rc in enumerate(point_cells):
        f = _nearest_free(g, rc)
        if f is not None:
            pts[i] = to_world(f)
    return {"grid": g, "cell": float(cell), "origin": (float(ox), float(oy)),
            "dock": to_world(dock_cell), "points": pts, "kind": kind}


def _from_existing(m, kind):
    """Normalise a build_procedural / apartment map dict to our save format."""
    return {"grid": m["grid"].astype(np.uint8), "cell": float(m["cell"]),
            "origin": (float(m["origin"][0]), float(m["origin"][1])),
            "dock": (float(m["dock"][0]), float(m["dock"][1])),
            "points": {int(k): (float(v[0]), float(v[1])) for k, v in m["points"].items()},
            "kind": kind}


def generate(n=1000, seed=0, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    index = []
    for i in range(n):
        r = rng.random()
        try:
            if r < 0.62:
                m = _from_existing(pd.build_procedural(rng), "procedural")
            elif r < 0.90:
                m = build_hotel(rng)
            else:                                    # apartment archetype (fixed shape)
                m = _from_existing(ac.build(), "apartment")
        except Exception:
            m = _from_existing(pd.build_procedural(rng), "procedural")
        # need a usable map: some free space + at least 1 delivery point
        if m["grid"].sum() < 30 or len(m["points"]) == 0:
            m = _from_existing(pd.build_procedural(rng), "procedural")
        fn = f"map_{i:04d}.npy"
        np.save(os.path.join(out_dir, fn), m["grid"])
        index.append({"file": fn, "kind": m["kind"], "cell": m["cell"],
                      "origin": list(m["origin"]), "dock": list(m["dock"]),
                      "points": {str(k): list(v) for k, v in m["points"].items()}})
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f)
    kinds = {}
    for e in index:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    return {"n": len(index), "kinds": kinds, "out_dir": out_dir}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    res = generate(args.n, args.seed)
    print(f"Đã sinh {res['n']} map -> {res['out_dir']}")
    print("Phân bố archetype:", res["kinds"])
