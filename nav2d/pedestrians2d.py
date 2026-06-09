"""Dynamic pedestrians for the 2D delivery env (Stage 3).

People are slow-moving circles that pace **inside the corridors only**. Validity
is enforced against the map's free-space (an occupancy grid of the static walls):
before a pedestrian moves, the new position is checked with ``is_free``; if it
would hit a wall or leave the building outline, the agent **bounces / picks a new
waypoint** in free space instead of walking through the wall.

They are visible to the robot's LiDAR (ray-vs-circle, merged with the wall
raycast). Speed is a realistic walking pace (~0.5-1.0 m/s); with the env control
step dt the per-step move is small.
"""

from __future__ import annotations

import math
import numpy as np


def _build_freegrid(world, cell=0.25):
    """Rasterise the static map into a free/occupied grid (1=free corridor).

    A cell is free if a robot-sized clearance around its centre hits no wall
    segment -> pedestrians then live strictly inside the walkable corridor."""
    xmin, xmax, ymin, ymax = world.bounds
    nrows = max(1, int(math.ceil((ymax - ymin) / cell)))
    ncols = max(1, int(math.ceil((xmax - xmin) / cell)))
    grid = np.zeros((nrows, ncols), dtype=np.uint8)
    for r in range(nrows):
        for c in range(ncols):
            x = xmin + (c + 0.5) * cell
            y = ymin + (r + 0.5) * cell
            if not world.segment_hits_circle((x, y), 0.30):
                grid[r, c] = 1
    return grid, cell, (xmin, ymin)


class Pedestrians:
    def __init__(self, world, rng, n=3, radius=0.28, speed_range=(0.5, 1.0),
                 dt=0.1, grid_map=None):
        self.world = world
        self.rng = rng
        self.radius = radius
        self.dt = dt
        # occupancy grid of free space (reuse a provided one, else rasterise)
        if grid_map is not None and grid_map.get("grid") is not None:
            self.grid = grid_map["grid"]; self.cell = grid_map["cell"]
            self.origin = grid_map["origin"]
        else:
            self.grid, self.cell, self.origin = _build_freegrid(world)
        self._free_cells = list(zip(*np.where(self.grid == 1)))
        self.pos = np.zeros((0, 2)); self.vel = np.zeros((0, 2))
        self._spawn(n, speed_range)

    # ---- free-space test on the occupancy grid -------------------------- #
    def is_free(self, x, y):
        ox, oy = self.origin
        c = int((x - ox) / self.cell); r = int((y - oy) / self.cell)
        if 0 <= r < self.grid.shape[0] and 0 <= c < self.grid.shape[1]:
            return self.grid[r, c] == 1
        return False

    def _cell_center(self, rc):
        ox, oy = self.origin
        return (ox + (rc[1] + 0.5) * self.cell, oy + (rc[0] + 0.5) * self.cell)

    def _spawn(self, n, speed_range):
        pos = []; vel = []
        if not self._free_cells:
            self.pos = np.zeros((0, 2)); self.vel = np.zeros((0, 2)); return
        tries = 0
        while len(pos) < n and tries < 3000:
            tries += 1
            rc = self._free_cells[self.rng.integers(len(self._free_cells))]
            p = np.array(self._cell_center(rc))
            if self.is_free(*p):
                ang = self.rng.uniform(0, 2 * math.pi)
                sp = self.rng.uniform(*speed_range)
                pos.append(p); vel.append([math.cos(ang) * sp, math.sin(ang) * sp])
        self.pos = np.array(pos) if pos else np.zeros((0, 2))
        self.vel = np.array(vel) if vel else np.zeros((0, 2))

    # ---- motion: walk, but NEVER through a wall ------------------------- #
    def step(self, dt=None):
        dt = self.dt if dt is None else dt
        for i in range(len(self.pos)):
            npos = self.pos[i] + self.vel[i] * dt
            # accept only if BOTH the new centre and a small clearance are free
            if self.is_free(npos[0], npos[1]) and \
               not self.world.segment_hits_circle(tuple(npos), self.radius):
                self.pos[i] = npos
            else:
                # blocked -> bounce: reverse + random jitter, retry a couple of
                # headings so the agent keeps moving but stays inside the corridor
                placed = False
                for _ in range(6):
                    ang = self.rng.uniform(0, 2 * math.pi)
                    sp = float(np.linalg.norm(self.vel[i])) or 0.6
                    cand_v = np.array([math.cos(ang) * sp, math.sin(ang) * sp])
                    cand = self.pos[i] + cand_v * dt
                    if self.is_free(cand[0], cand[1]) and \
                       not self.world.segment_hits_circle(tuple(cand), self.radius):
                        self.vel[i] = cand_v; self.pos[i] = cand; placed = True; break
                if not placed:
                    self.vel[i] = -self.vel[i]   # last resort: just reverse, stay put

    # ---- sensing: ray-vs-circle distances merged into a LiDAR fan -------- #
    def raycast_into(self, origin, angles, ranges, max_range):
        if len(self.pos) == 0:
            return ranges
        ox, oy = origin
        dx = np.cos(angles); dy = np.sin(angles)
        out = ranges.copy()
        for (cx, cy) in self.pos:
            r = self.radius
            fx = cx - ox; fy = cy - oy
            tca = fx * dx + fy * dy
            d2 = (fx * fx + fy * fy) - tca * tca
            hit = (tca > 0) & (d2 <= r * r)
            thc = np.sqrt(np.clip(r * r - d2, 0, None))
            thit = tca - thc
            cand = np.where(hit & (thit >= 0) & (thit < out), thit, out)
            out = np.minimum(out, cand)
        return np.clip(out, 0.0, max_range).astype(np.float32)

    def hits_robot(self, p, robot_radius):
        if len(self.pos) == 0:
            return False
        d = np.linalg.norm(self.pos - np.array(p), axis=1)
        return bool(np.any(d <= self.radius + robot_radius))
