"""Training env: learn to FOLLOW a planned route on the fixed delivery maps.

This is the env that fixes the "weaving / doesn't arrive" problem. Each episode:
  * load a fixed map (apartment_a / b), pick a random delivery point,
  * the global A* planner produces the route dock->point (a polyline),
  * the robot must drive along it using ONLY LiDAR + the relative bearing to a
    **pure-pursuit lookahead point** on that route (the same signal used at
    deploy time), reaching the goal smoothly.

Reward is designed for STEADY driving:
  + progress measured ALONG the route (not straight-line) -> rewards following,
  + arrival bonus,
  - collision, - time, - turn magnitude AND - turn *changes* (anti-weave),
  - cross-track error (distance off the planned path) -> stay on the line.

Observation matches the deploy controller exactly:
  [ n_lidar | sin(bearing_to_lookahead), cos, dist_norm | cross_track_norm
  | prev_turn | fwd_clear ]
Action: [forward, turn]  (same differential drive as nav_env).
"""

from __future__ import annotations

import math
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from fixed_maps import build_map, map_names
from grid_planner import GridPlanner


class DeliveryFollowEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 20}

    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        c = config or {}
        self.map_names = c.get("maps", list(map_names()))
        self.maps = {n: build_map(n) for n in self.map_names}
        self.planners = {n: GridPlanner(self.maps[n], inflate=0) for n in self.map_names}
        self.planners_inf = {n: GridPlanner(self.maps[n], inflate=1) for n in self.map_names}

        self.robot_radius = 0.22
        self.max_speed = float(c.get("max_speed", 0.9))
        self.max_turn = float(c.get("max_turn", 2.2))
        self.dt = float(c.get("dt", 0.1))
        self.n_lidar = int(c.get("n_lidar", 24))
        self.lidar_range = float(c.get("lidar_range", 5.0))
        self.lidar_angles = np.linspace(0, 2 * math.pi, self.n_lidar, endpoint=False)
        self.lookahead = float(c.get("lookahead", 1.6))
        self.max_steps = int(c.get("max_steps", 700))
        self.arrive_dist = float(c.get("arrive_dist", 0.5))

        # reward weights
        self.w_progress = float(c.get("w_progress", 1.5))
        self.w_arrive = float(c.get("w_arrive", 40.0))
        self.w_collide = float(c.get("w_collide", 10.0))
        self.w_time = float(c.get("w_time", 0.01))
        self.w_turn = float(c.get("w_turn", 0.03))
        self.w_dturn = float(c.get("w_dturn", 0.05))     # anti-weave (turn change)
        self.w_xtrack = float(c.get("w_xtrack", 0.4))     # stay on the line
        self.collision_grace = int(c.get("collision_grace", 20))

        obs_dim = self.n_lidar + 3 + 1 + 1 + 1
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.rng = np.random.default_rng(0)
        self._init_state()

    def _init_state(self):
        self.world = None; self.path = []; self.seg_i = 0
        self.pos = np.zeros(2); self.heading = 0.0; self.prev_turn = 0.0
        self.goal = np.zeros(2); self.step_i = 0; self.bump = 0
        self.prev_along = 0.0; self.path_len = 0.0

    # ------------------------------------------------------------------ #
    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        pid = 0
        for _try in range(12):
            name = (options or {}).get("map") or self.rng.choice(self.map_names)
            m = self.maps[name]
            # Start AND goal are any of {dock, points} -> the policy learns to drive
            # between ANY two locations, which crucially includes the RETURN leg
            # (point -> dock), not just dock -> point. Fixes "arrives but can't get
            # back" seen with one-way-only training.
            nodes = [("dock", m["dock"])] + [(p, m["points"][p]) for p in m["points"]]
            opt_pt = (options or {}).get("point")
            if opt_pt is not None:
                s_idx = 0; g_node = (opt_pt, m["points"][opt_pt])  # dock->chosen point
                start = m["dock"]; goal = g_node[1]; pid = opt_pt
            else:
                i, j = self.rng.choice(len(nodes), size=2, replace=False)
                start = nodes[i][1]; goal = nodes[j][1]
                pid = nodes[j][0] if nodes[j][0] != "dock" else nodes[i][0]
            wps, L = self.planners_inf[name].plan(start, goal)
            if wps is None:
                wps, L = self.planners[name].plan(start, goal)
            if wps and L > 2.0:
                self.world = m["world"]; self.path = list(wps); self.path_len = L
                break
        self._init_state_from_path(start, goal)
        return self._obs(), {"map": name, "point": pid}

    def _init_state_from_path(self, start, goal):
        self.pos = np.array(start, dtype=np.float64)
        self.goal = np.array(goal, dtype=np.float64)
        d = np.array(self.path[1]) - np.array(self.path[0]) if len(self.path) > 1 else np.array([1., 0.])
        self.heading = math.atan2(d[1], d[0])
        self.prev_turn = 0.0; self.step_i = 0; self.bump = 0; self.seg_i = 0
        self.prev_along = 0.0

    # ---- pure-pursuit + cross-track helpers --------------------------- #
    def _project_along(self):
        """Return (distance travelled ALONG path, cross-track distance to path)."""
        p = self.pos; best_d = 1e9; along = 0.0; acc = 0.0; best_along = 0.0
        for j in range(len(self.path) - 1):
            a = np.array(self.path[j]); b = np.array(self.path[j + 1])
            ab = b - a; L = float(np.linalg.norm(ab)) or 1e-6
            t = float(np.clip(np.dot(p - a, ab) / (L * L), 0, 1))
            proj = a + ab * t; d = float(np.linalg.norm(p - proj))
            if d < best_d:
                best_d = d; best_along = acc + t * L
            acc += L
        return best_along, best_d

    def _lookahead_point(self):
        while self.seg_i < len(self.path) - 1:
            if np.linalg.norm(np.array(self.path[self.seg_i]) - self.pos) < self.lookahead * 0.6:
                self.seg_i += 1
            else:
                break
        acc = 0.0; prev = self.pos.copy()
        for j in range(self.seg_i, len(self.path)):
            q = np.array(self.path[j]); d = float(np.linalg.norm(q - prev))
            if acc + d >= self.lookahead:
                t = (self.lookahead - acc) / max(d, 1e-6)
                return prev + (q - prev) * t
            acc += d; prev = q
        return np.array(self.path[-1])

    def _lidar(self):
        rngs = np.empty(self.n_lidar, dtype=np.float32)
        for i, a in enumerate(self.lidar_angles):
            rngs[i] = self.world.raycast(tuple(self.pos), self.heading + a, self.lidar_range)
        return np.clip(rngs / self.lidar_range, 0.0, 1.0)

    def _obs(self, lidar=None):
        lidar = self._lidar() if lidar is None else lidar
        look = self._lookahead_point()
        d = look - self.pos
        bearing = (math.atan2(d[1], d[0]) - self.heading + math.pi) % (2 * math.pi) - math.pi
        gdist = float(np.linalg.norm(self.goal - self.pos))
        _, xtrack = self._project_along()
        obs = np.concatenate([
            lidar,
            [math.sin(bearing), math.cos(bearing), min(gdist / 20.0, 1.0)],
            [float(np.clip(xtrack / 1.0, 0, 1))],
            [self.prev_turn / self.max_turn],
            [lidar[0]],
        ]).astype(np.float32)
        return np.clip(obs, -1.0, 1.0)

    # ------------------------------------------------------------------ #
    def step(self, action):
        self.step_i += 1
        a = np.clip(np.asarray(action, np.float32), -1, 1)
        fwd = (a[0] * 0.5 + 0.5) * self.max_speed
        turn = a[1] * self.max_turn
        dturn = abs(turn - self.prev_turn)

        nh = (self.heading + turn * self.dt + math.pi) % (2 * math.pi) - math.pi
        npos = self.pos + np.array([math.cos(nh), math.sin(nh)]) * fwd * self.dt
        collided = self.world.segment_hits_circle(tuple(npos), self.robot_radius)
        if not collided:
            self.pos = npos; self.bump = 0
        else:
            self.bump += 1
        self.heading = nh

        along, xtrack = self._project_along()
        reward = self.w_progress * (along - self.prev_along)
        self.prev_along = along
        reward -= self.w_time
        reward -= self.w_turn * abs(a[1])
        reward -= self.w_dturn * dturn
        reward -= self.w_xtrack * min(xtrack, 1.0)

        terminated = False
        info = {"collision": False, "arrived": False}
        gdist = float(np.linalg.norm(self.goal - self.pos))
        if collided:
            reward -= self.w_collide
            if self.bump > self.collision_grace:
                terminated = True; info["collision"] = True
        if gdist <= self.arrive_dist:
            reward += self.w_arrive; terminated = True; info["arrived"] = True
        truncated = self.step_i >= self.max_steps
        self.prev_turn = turn
        return self._obs(), float(reward), terminated, truncated, info
