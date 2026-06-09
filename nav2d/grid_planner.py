"""A* grid path planner for the fixed delivery maps.

Works directly on the occupancy grid from ``fixed_maps.build_map`` (1 = free).
Provides:
  * shortest grid path between two world points (8-connected A*),
  * its real path length (metres) — used by the TSP route optimizer for
    "tiện đường" (true travel cost, not straight-line),
  * waypoint simplification via line-of-sight string-pulling, so the hybrid
    runner steers through a few corner waypoints instead of every cell.
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Tuple

import numpy as np


class GridPlanner:
    def __init__(self, map_dict: Dict, inflate: int = 0):
        self.grid = map_dict["grid"]
        self.cell = map_dict["cell"]
        self.ox, self.oy = map_dict["origin"]
        self.R, self.C = self.grid.shape
        self.inflate = inflate

    # ---- coordinate transforms ---------------------------------------- #
    def to_cell(self, xy) -> Tuple[int, int]:
        c = int(round((xy[0] - self.ox) / self.cell - 0.5))
        r = int(round((xy[1] - self.oy) / self.cell - 0.5))
        return (r, c)

    def to_world(self, rc) -> Tuple[float, float]:
        r, c = rc
        return (self.ox + (c + 0.5) * self.cell, self.oy + (r + 0.5) * self.cell)

    def _free(self, r, c) -> bool:
        if not (0 <= r < self.R and 0 <= c < self.C):
            return False
        if self.grid[r, c] != 1:
            return False
        if self.inflate:
            for dr in range(-self.inflate, self.inflate + 1):
                for dc in range(-self.inflate, self.inflate + 1):
                    rr, cc = r + dr, c + dc
                    if not (0 <= rr < self.R and 0 <= cc < self.C and self.grid[rr, cc] == 1):
                        return False
        return True

    def _nearest_free(self, rc, rad=6):
        if self._free(*rc):
            return rc
        for k in range(1, rad + 1):
            for dr in range(-k, k + 1):
                for dc in range(-k, k + 1):
                    cand = (rc[0] + dr, rc[1] + dc)
                    if self._free(*cand):
                        return cand
        return None

    # ---- A* ----------------------------------------------------------- #
    def _astar(self, s, g) -> Optional[List[Tuple[int, int]]]:
        s = self._nearest_free(s); g = self._nearest_free(g)
        if s is None or g is None:
            return None
        nb = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
              (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
        def h(a): return math.hypot(a[0] - g[0], a[1] - g[1])
        openq = [(h(s), 0.0, s)]
        gscore = {s: 0.0}; came = {s: None}
        while openq:
            _, gc, cur = heapq.heappop(openq)
            if cur == g:
                path = []
                while cur is not None:
                    path.append(cur); cur = came[cur]
                return path[::-1]
            if gc > gscore.get(cur, 1e18):
                continue
            for dr, dc, w in nb:
                nr, nc = cur[0] + dr, cur[1] + dc
                if not self._free(nr, nc):
                    continue
                if dr != 0 and dc != 0:  # no corner cutting
                    if not (self._free(cur[0] + dr, cur[1]) and self._free(cur[0], cur[1] + dc)):
                        continue
                ng = gc + w
                if ng < gscore.get((nr, nc), 1e18):
                    gscore[(nr, nc)] = ng; came[(nr, nc)] = cur
                    heapq.heappush(openq, (ng + h((nr, nc)), ng, (nr, nc)))
        return None

    def _line_free(self, a, b) -> bool:
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(2, int(d * 2))
        for k in range(n + 1):
            t = k / n
            r = int(round(a[0] + (b[0] - a[0]) * t))
            c = int(round(a[1] + (b[1] - a[1]) * t))
            if not self._free(r, c):
                return False
        return True

    def _simplify(self, cells) -> List[Tuple[int, int]]:
        if len(cells) <= 2:
            return cells
        wp = [cells[0]]; anchor = 0; i = 1
        while i < len(cells) - 1:
            if self._line_free(cells[anchor], cells[i + 1]):
                i += 1
            else:
                wp.append(cells[i]); anchor = i; i += 1
        wp.append(cells[-1])
        return wp

    # ---- public ------------------------------------------------------- #
    def plan(self, start_xy, goal_xy):
        """Return (waypoints_world, path_length_m) or (None, inf)."""
        cells = self._astar(self.to_cell(start_xy), self.to_cell(goal_xy))
        if not cells:
            return None, float("inf")
        length = 0.0
        for a, b in zip(cells[:-1], cells[1:]):
            length += math.hypot((a[0]-b[0]) * self.cell, (a[1]-b[1]) * self.cell)
        wps = [self.to_world(c) for c in self._simplify(cells)]
        wps[0] = tuple(start_xy); wps[-1] = tuple(goal_xy)
        return wps, length

    def distance(self, a_xy, b_xy) -> float:
        _, L = self.plan(a_xy, b_xy)
        return L
