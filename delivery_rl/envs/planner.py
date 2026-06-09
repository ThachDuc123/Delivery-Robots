"""Grid path planner (global) -> waypoints for the local RL controller.

The RL policy is purely reactive (lidar + relative pose to a target), so it
cannot see around corners. For a rich map (rooms, an arc branch) we add the
standard "global planner + local controller" split: BFS on a coarse occupancy
grid finds a path from the robot to the destination locker dock, which is then
reduced to a short list of WAYPOINTS. The env feeds the *next* waypoint as the
policy's relative-pose target, so the policy follows the corridor around bends
while still doing all obstacle avoidance from its sensors.

Occupancy comes from ``CorridorWorld.is_free`` (geometry-based), inflated by the
robot radius so paths keep clearance from walls.
"""

from __future__ import annotations

import math
from collections import deque
from typing import List, Optional, Tuple

import numpy as np


class GridPlanner:
    def __init__(self, world, robot_radius: float, cell: float = 0.25):
        self.world = world
        self.cell = cell
        self.clearance = robot_radius + 0.04
        xmin, xmax, ymin, ymax = world.scene.bounds
        self.xmin, self.ymin = xmin, ymin
        self.ncols = max(1, int((xmax - xmin) / cell))
        self.nrows = max(1, int((ymax - ymin) / cell))

    def _to_cell(self, x, y) -> Tuple[int, int]:
        return (int((x - self.xmin) / self.cell), int((y - self.ymin) / self.cell))

    def _to_world(self, cx, cy) -> Tuple[float, float]:
        return (self.xmin + (cx + 0.5) * self.cell, self.ymin + (cy + 0.5) * self.cell)

    def _free(self, cx, cy) -> bool:
        if not (0 <= cx < self.ncols and 0 <= cy < self.nrows):
            return False
        wx, wy = self._to_world(cx, cy)
        return self.world.is_free(wx, wy, self.clearance)

    def plan(self, start_xy, goal_xy) -> Optional[List[Tuple[float, float]]]:
        """Return a list of world-space waypoints from start to goal, or None."""
        s = self._to_cell(*start_xy)
        g = self._to_cell(*goal_xy)
        s = self._nearest_free(s)
        g = self._nearest_free(g)
        if s is None or g is None:
            return None
        # BFS (8-connected) with parent tracking
        seen = {s}
        parent = {s: None}
        q = deque([s])
        neigh = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        found = False
        while q:
            cur = q.popleft()
            if cur == g:
                found = True
                break
            for dx, dy in neigh:
                nb = (cur[0] + dx, cur[1] + dy)
                if nb in seen or not self._free(*nb):
                    continue
                # avoid cutting diagonal corners
                if dx != 0 and dy != 0:
                    if not (self._free(cur[0] + dx, cur[1]) and self._free(cur[0], cur[1] + dy)):
                        continue
                seen.add(nb)
                parent[nb] = cur
                q.append(nb)
        if not found:
            return None
        # reconstruct
        cells = []
        cur = g
        while cur is not None:
            cells.append(cur)
            cur = parent[cur]
        cells.reverse()
        pts = [self._to_world(*c) for c in cells]
        pts[0] = tuple(start_xy)
        pts[-1] = tuple(goal_xy)
        return self._simplify_los(pts)

    def _nearest_free(self, cell, radius=6) -> Optional[Tuple[int, int]]:
        if self._free(*cell):
            return cell
        for r in range(1, radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    c = (cell[0] + dx, cell[1] + dy)
                    if self._free(*c):
                        return c
        return None

    def _line_free(self, a, b) -> bool:
        """True if the straight segment a->b stays in free space (sampled)."""
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(2, int(d / (self.cell * 0.5)))
        for k in range(n + 1):
            t = k / n
            x = a[0] + (b[0] - a[0]) * t
            y = a[1] + (b[1] - a[1]) * t
            if not self.world.is_free(x, y, self.clearance):
                return False
        return True

    def _simplify_los(self, pts) -> List[Tuple[float, float]]:
        """String-pulling: keep only the waypoints needed so each consecutive
        pair has line-of-sight in free space. This preserves corner points at
        room mouths / the arc (where the path actually bends) instead of cutting
        across walls, which the previous fixed-stride simplifier did."""
        if len(pts) <= 2:
            return [tuple(map(float, p)) for p in pts]
        wp = [pts[0]]
        anchor = 0
        i = 1
        while i < len(pts) - 1:
            if self._line_free(pts[anchor], pts[i + 1]):
                i += 1  # can still see the next one -> skip current
            else:
                wp.append(pts[i]); anchor = i; i += 1
        wp.append(pts[-1])
        return [tuple(map(float, p)) for p in wp]
