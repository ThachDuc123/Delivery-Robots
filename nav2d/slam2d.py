"""2D SLAM + Frontier Exploration (blind mapping) for the delivery robot.

The robot starts with **no map**. It carries a 2D LiDAR; as it drives, each scan
is integrated into an **occupancy grid** (log-odds): cells the rays pass through
become FREE, the cell at each hit becomes OCCUPIED, everything else stays UNKNOWN.
**Frontier exploration** repeatedly drives the robot to the nearest boundary
between known-free and unknown space, expanding the map until the reachable area
is covered. The finished occupancy grid is then handed to the global planner
(A* + TSP) for delivery — Stage 2.2.

This is a simulation-grade Gmapping/Cartographer + frontier-explorer analogue:
pose is taken from the (noisy) simulator odometry; the focus is the mapping +
exploration logic, not full scan-matching SLAM.

Public:
  OccupancyGrid          : the live map (UNKNOWN/FREE/OCC), ray integration, queries
  FrontierExplorer.explore(world, start_xy) -> (grid, pose_trail, frames_meta)
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

UNKNOWN, FREE, OCC = -1, 0, 1


class OccupancyGrid:
    def __init__(self, bounds, cell=0.3):
        xmin, xmax, ymin, ymax = bounds
        self.cell = cell
        self.xmin, self.ymin = xmin, ymin
        self.ncols = max(1, int(math.ceil((xmax - xmin) / cell)))
        self.nrows = max(1, int(math.ceil((ymax - ymin) / cell)))
        # log-odds accumulator; sign -> FREE/OCC, near-0 -> UNKNOWN
        self.logodds = np.zeros((self.nrows, self.ncols), dtype=np.float32)
        self.seen = np.zeros((self.nrows, self.ncols), dtype=bool)
        self.l_free, self.l_occ, self.clamp = -0.4, 0.85, 6.0

    def to_cell(self, x, y):
        return (int((y - self.ymin) / self.cell), int((x - self.xmin) / self.cell))

    def to_world(self, r, c):
        return (self.xmin + (c + 0.5) * self.cell, self.ymin + (r + 0.5) * self.cell)

    def in_bounds(self, r, c):
        return 0 <= r < self.nrows and 0 <= c < self.ncols

    def state(self, r, c):
        if not self.in_bounds(r, c) or not self.seen[r, c]:
            return UNKNOWN
        return OCC if self.logodds[r, c] > 0.3 else FREE

    def _bump(self, r, c, d):
        if self.in_bounds(r, c):
            self.logodds[r, c] = float(np.clip(self.logodds[r, c] + d, -self.clamp, self.clamp))
            self.seen[r, c] = True

    def integrate_scan(self, pose_xy, yaw, ranges, angles, max_range):
        """Bresenham-ish ray integration of one LiDAR scan into the grid."""
        ox, oy = pose_xy
        r0, c0 = self.to_cell(ox, oy)
        for rng, a in zip(ranges, angles):
            ang = yaw + a
            hit = rng < max_range - 1e-3
            ex = ox + math.cos(ang) * rng
            ey = oy + math.sin(ang) * rng
            r1, c1 = self.to_cell(ex, ey)
            # mark free cells along the ray (exclude endpoint if it's a hit)
            for (rr, cc) in self._line(r0, c0, r1, c1)[:-1]:
                self._bump(rr, cc, self.l_free)
            if hit:
                self._bump(r1, c1, self.l_occ)
            else:
                self._bump(r1, c1, self.l_free)

    @staticmethod
    def _line(r0, c0, r1, c1):
        pts = []
        dr = abs(r1 - r0); dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1; sc = 1 if c0 < c1 else -1
        err = dr - dc; r, c = r0, c0
        while True:
            pts.append((r, c))
            if r == r1 and c == c1:
                break
            e2 = 2 * err
            if e2 > -dc:
                err -= dc; r += sr
            if e2 < dr:
                err += dr; c += sc
            if len(pts) > 4000:
                break
        return pts

    def free_mask(self):
        return self.seen & (self.logodds <= 0.3)

    def coverage_fraction(self, reachable_cells):
        """Fraction of the ground-truth reachable cells now mapped as FREE."""
        if not reachable_cells:
            return 1.0
        got = sum(1 for (r, c) in reachable_cells if self.state(r, c) == FREE)
        return got / len(reachable_cells)


class FrontierExplorer:
    def __init__(self, world, cell=0.3, lidar_n=36, lidar_range=5.0,
                 step=0.18, reach=0.35, max_steps=4000):
        self.world = world
        self.cell = cell
        self.lidar_n = lidar_n
        self.lidar_range = lidar_range
        self.lidar_angles = np.linspace(0, 2 * math.pi, lidar_n, endpoint=False)
        self.step = step           # metres per move step
        self.reach = reach         # waypoint reach tol
        self.max_steps = max_steps
        self.robot_radius = 0.22
        self.replan_every = 14     # re-detect frontiers every N cells along a leg

    # ---- sensing -------------------------------------------------------- #
    def _scan(self, pos, yaw):
        # vectorised full-fan raycast (numpy) -> the SLAM bottleneck, batched
        return self.world.raycast_batch(tuple(pos), yaw + self.lidar_angles, self.lidar_range)

    # ---- frontier detection -------------------------------------------- #
    def _frontiers(self, grid: OccupancyGrid):
        """Free cells that touch at least one UNKNOWN cell = exploration frontier."""
        fr = []
        free = grid.free_mask()
        for r in range(grid.nrows):
            for c in range(grid.ncols):
                if not free[r, c]:
                    continue
                for nr, nc in ((r+1, c), (r-1, c), (r, c+1), (r, c-1)):
                    if grid.in_bounds(nr, nc) and not grid.seen[nr, nc]:
                        fr.append((r, c)); break
        return fr

    # ---- A* on currently-known-free grid ------------------------------- #
    def _astar(self, grid, start_rc, goal_rc):
        def passable(r, c):
            return grid.state(r, c) == FREE
        if not passable(*goal_rc):
            return None
        openq = [(0, start_rc)]; came = {start_rc: None}; g = {start_rc: 0.0}
        while openq:
            _, cur = heapq.heappop(openq)
            if cur == goal_rc:
                path = []
                while cur: path.append(cur); cur = came[cur]
                return path[::-1]
            for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                nb = (cur[0]+dr, cur[1]+dc)
                if not grid.in_bounds(*nb) or not passable(*nb):
                    continue
                ng = g[cur] + 1
                if ng < g.get(nb, 1e18):
                    g[nb] = ng; came[nb] = cur
                    h = abs(nb[0]-goal_rc[0]) + abs(nb[1]-goal_rc[1])
                    heapq.heappush(openq, (ng + h, nb))
        return None

    def _nearest_frontier_path(self, grid, cur_rc, frontiers):
        """BFS from the robot over known-free cells to the closest frontier; return
        the path (list of cells) to it."""
        if not frontiers:
            return None
        fset = set(frontiers)
        seen = {cur_rc}; par = {cur_rc: None}; q = deque([cur_rc])
        while q:
            cur = q.popleft()
            if cur in fset and cur != cur_rc:
                path = []
                while cur is not None: path.append(cur); cur = par[cur]
                return path[::-1]
            for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                nb = (cur[0]+dr, cur[1]+dc)
                if grid.in_bounds(*nb) and nb not in seen and grid.state(*nb) == FREE:
                    seen.add(nb); par[nb] = cur; q.append(nb)
        return None

    # ---- main loop ------------------------------------------------------ #
    def explore(self, start_xy, reachable_cells=None, coverage_target=0.97,
                record=False) -> Dict:
        grid = OccupancyGrid(self.world.bounds, self.cell)
        pos = np.array(start_xy, dtype=np.float64)
        yaw = 0.0
        trail = [tuple(pos)]
        frames = []  # (pose, grid snapshot) for the GIF, sampled
        grid.integrate_scan(pos, yaw, self._scan(pos, yaw), self.lidar_angles, self.lidar_range)

        steps = 0
        blocked_frontiers = set()   # frontier cells we failed to make progress toward
        stall = 0
        while steps < self.max_steps:
            steps_before = steps
            frontiers = [f for f in self._frontiers(grid) if f not in blocked_frontiers]
            cur_rc = grid.to_cell(*pos)
            fpath = self._nearest_frontier_path(grid, cur_rc, frontiers)
            if fpath is None or len(fpath) < 2:
                break  # nothing left to explore that we can reach
            target_frontier = fpath[-1]
            # drive along the frontier path, integrating scans as we go.
            # Re-plan only every `replan_every` cells (frontier scan is costly);
            # coverage is checked occasionally, not every step.
            done_cov = False
            # Follow the BFS path CELL-BY-CELL (each step is to an adjacent known-
            # free cell centre) so the robot reliably reaches deep niches instead
            # of free-angle sliding past their mouths.
            for ci, cell_rc in enumerate(fpath[1:]):
                tgt = np.array(grid.to_world(*cell_rc))
                guard = 0
                # move toward this path cell in small increments, scanning each
                # increment, until within ~half a cell of it.
                while np.linalg.norm(tgt - pos) > self.cell * 0.5 and guard < 40 and steps < self.max_steps:
                    d = tgt - pos
                    yaw = math.atan2(d[1], d[0])
                    stepd = min(self.step, float(np.linalg.norm(d)))
                    npos = pos + np.array([math.cos(yaw), math.sin(yaw)]) * stepd
                    if self.world.segment_hits_circle(tuple(npos), self.robot_radius):
                        break               # blocked -> re-plan frontiers
                    pos = npos; trail.append(tuple(pos)); steps += 1; guard += 1
                    grid.integrate_scan(pos, yaw, self._scan(pos, yaw),
                                        self.lidar_angles, self.lidar_range)
                    if record and steps % 8 == 0:
                        frames.append((tuple(pos), grid.logodds.copy(), grid.seen.copy()))
                    if (reachable_cells is not None and steps % 25 == 0
                            and grid.coverage_fraction(reachable_cells) >= coverage_target):
                        done_cov = True; break
                if done_cov or ci + 1 >= self.replan_every:
                    break   # re-plan frontiers from the new vantage point
            if done_cov:
                break
            # progress guard: if this whole outer iteration moved the robot < 1
            # step, the chosen frontier is unreachable -> blacklist it & retry; if
            # we stall repeatedly, exploration is finished.
            if steps - steps_before < 1:
                blocked_frontiers.add(target_frontier)
                stall += 1
                if stall > 30:
                    break
            else:
                stall = 0
        cov = grid.coverage_fraction(reachable_cells) if reachable_cells is not None else None
        return {"grid": grid, "trail": trail, "steps": steps, "coverage": cov,
                "frames": frames, "final_pose": tuple(pos)}
