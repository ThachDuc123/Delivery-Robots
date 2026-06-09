"""Multi-stop delivery env (Train == Deploy) for the fixed maps.

One episode = one full delivery trip on a fixed apartment map:
    dock -> P_a -> P_b -> P_c -> dock
The **Global Planner (A* + TSP)** lives inside ``reset()``: it picks 1-3 random
delivery points, orders them optimally, and lays the whole waypoint route. The
robot follows it with a pure-pursuit lookahead, using **only LiDAR + ego-centric
bearing to the lookahead** (no absolute coordinates). When it reaches a stop it
does NOT reset — the env pops the stop and switches the lookahead to the next leg
(**Continuous Goal Transition**), so the policy learns the hard "spin around in a
dead-end and chase the new target behind me" moment directly.

Design decisions (agreed with the user):
  * **Grace period**: for a few steps after each stop transition, cross-track and
    turn penalties are suppressed so the robot can freely turn/back out of a niche
    without being punished for the 180-degree lookahead flip.
  * **Collision-grace**: a light bump just holds position + small penalty; only a
    persistent jam ends the episode (a tight-niche U-turn needs to graze walls).
  * **Reverse allowed**: forward action maps to [-0.4*max, +max] so the robot can
    back out of dead ends instead of being forced into an in-place U-turn.
  * **Tiered rewards**: progress along route, hot bonus per intermediate stop,
    bigger bonus for returning to the dock (episode end).

Because this env IS the deployment loop, evaluation just runs the policy in it —
no external hybrid_runner, so there is no train/deploy mismatch.

Observation (ego-centric, all ~[-1,1]):
    [ n_lidar | sin(bearing_to_lookahead), cos, goal_dist_norm
    | cross_track_norm | prev_turn | front_clear | grace_flag
    | stops_left_norm ]
Action: [forward (-0.4..1 of max), turn].
"""

from __future__ import annotations

import math
import os
from collections import deque
from typing import List, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from fixed_maps import build_map, map_names
from grid_planner import GridPlanner
from delivery_planner import DeliveryPlanner


class MultiStopEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 20}

    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        c = config or {}
        # "procedural": a NEW random map every reset (generalization training);
        # otherwise use the named fixed maps (a/b/c) -- used for held-out eval.
        self.procedural = bool(c.get("procedural", False))
        self.fixed_mix = float(c.get("fixed_mix", 0.0))   # frac of episodes using apartment_complex
        # Robust training: sample from a pool of pre-generated maps in `map_dir`
        # (data/maps/ with index.json). Each reset loads one saved occupancy grid.
        self.map_dir = c.get("map_dir", None)
        self._map_index = None
        if self.map_dir:
            import json as _json
            with open(os.path.join(self.map_dir, "index.json"), encoding="utf-8") as _f:
                self._map_index = _json.load(_f)
        self.map_names = c.get("maps", list(map_names()))
        if not self.procedural:
            self.maps = {n: build_map(n) for n in self.map_names}
            self.dps = {n: DeliveryPlanner(self.maps[n]) for n in self.map_names}
            self.planners_inf = {n: GridPlanner(self.maps[n], inflate=1) for n in self.map_names}
            self.planners = {n: GridPlanner(self.maps[n], inflate=0) for n in self.map_names}

        # domain randomization (Level-2): jitter sensing + actuation each episode
        self.dr = bool(c.get("domain_random", self.procedural))
        self.lidar_noise_max = float(c.get("lidar_noise_max", 0.02))
        self.speed_jitter = float(c.get("speed_jitter", 0.15))   # +-15% wheel gain
        self.turn_jitter = float(c.get("turn_jitter", 0.15))
        self._lidar_noise = 0.0; self._speed_mult = 1.0; self._turn_mult = 1.0

        self.robot_radius = 0.22
        self.max_speed = float(c.get("max_speed", 0.9))
        self.max_turn = float(c.get("max_turn", 2.2))
        self.reverse_frac = float(c.get("reverse_frac", 0.4))     # how much reverse allowed
        self.dt = float(c.get("dt", 0.1))
        self.n_lidar = int(c.get("n_lidar", 24))
        self.lidar_range = float(c.get("lidar_range", 5.0))
        self.lidar_angles = np.linspace(0, 2 * math.pi, self.n_lidar, endpoint=False)
        self.lookahead = float(c.get("lookahead", 1.6))
        self.arrive_dist = float(c.get("arrive_dist", 0.5))
        self.max_steps = int(c.get("max_steps", 2000))           # long horizon: full trip
        self.min_parcels = int(c.get("min_parcels", 1))
        self.max_parcels = int(c.get("max_parcels", 3))

        # grace + collision config
        self.grace_steps_on_switch = int(c.get("grace_steps", 18))
        self.collision_grace = int(c.get("collision_grace", 25))

        # reward weights (tiered)
        self.w_progress = float(c.get("w_progress", 1.5))
        self.w_stop = float(c.get("w_stop", 50.0))
        self.w_dock = float(c.get("w_dock", 100.0))
        self.w_collide = float(c.get("w_collide", 8.0))
        self.w_jam = float(c.get("w_jam", 60.0))
        self.w_time = float(c.get("w_time", 0.01))
        self.w_turn = float(c.get("w_turn", 0.03))
        self.w_dturn = float(c.get("w_dturn", 0.05))
        self.w_xtrack = float(c.get("w_xtrack", 0.4))
        # wall-clearance + straight-driving shaping (robust "đi đẹp" reward):
        #  - penalise getting too close to a wall (front cone clearance < safe_dist)
        #  - reward driving straight & steady (low |turn|, low steering change) in
        #    the clear -> "đi thẳng ổn định, không zigzag".
        self.w_near_wall = float(c.get("w_near_wall", 0.6))
        self.safe_dist = float(c.get("safe_dist", 0.55))         # m
        self.w_straight = float(c.get("w_straight", 0.06))
        self.shaping_clip = c.get("shaping_clip", None)          # clip dense shaping (None=off)
        if self.shaping_clip is not None:
            self.shaping_clip = float(self.shaping_clip)
        # stuck-penalty: if the robot lingers inside a small radius for too many
        # steps (ramming a wall / spinning in place), penalise it so the policy
        # learns to widen its search / back out instead of head-butting bricks.
        self.w_stuck = float(c.get("w_stuck", 0.4))
        self.stuck_radius = float(c.get("stuck_radius", 0.4))     # m
        self.stuck_window = int(c.get("stuck_window", 40))        # steps

        # Stage 3: dynamic pedestrians + LiDAR frame-stacking. With n_ped>0 the
        # observation stacks the last `lidar_stack` LiDAR fans so the policy can
        # read each person's motion (velocity) and dodge proactively.
        self.n_ped = int(c.get("n_ped", 0))
        self.ped_speed = tuple(c.get("ped_speed", (0.5, 1.0)))  # realistic walking pace m/s
        self.lidar_stack = int(c.get("lidar_stack", 3)) if self.n_ped > 0 else 1
        self.w_ped_hit = float(c.get("w_ped_hit", 12.0))
        self.peds = None

        extra = self.n_lidar * (self.lidar_stack - 1)   # stacked older lidar fans
        obs_dim = self.n_lidar + 3 + 1 + 1 + 1 + 1 + 1 + extra
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.rng = np.random.default_rng(0)
        self._reset_vars()

    def _reset_vars(self):
        self.world = None; self.legs = []; self.leg_i = 0
        self.path = []; self.seg_i = 0
        self.pos = np.zeros(2); self.heading = 0.0; self.prev_turn = 0.0
        self.step_i = 0; self.bump = 0; self.grace = 0
        self.prev_along = 0.0; self.stops_total = 1; self.stops_done = 0
        self._pos_hist = deque(maxlen=getattr(self, "stuck_window", 40))
        self._lidar_hist = deque(maxlen=getattr(self, "lidar_stack", 1))

    # ------------------------------------------------------------------ #
    def _route_leg(self, name, a, b):
        wps, L = self.planners_inf[name].plan(a, b)
        if wps is None:
            wps, L = self.planners[name].plan(a, b)
        return wps, L

    def _load_saved_map(self, entry):
        """Build a full map dict from a saved index entry (grid .npy + meta)."""
        from world2d import World
        grid = np.load(os.path.join(self.map_dir, entry["file"])).astype(np.uint8)
        cell = float(entry["cell"]); ox, oy = entry["origin"]
        R, C = grid.shape
        def wall(r, c): return not (0 <= r < R and 0 <= c < C and grid[r, c] == 1)
        segs = []
        for r in range(R):
            for c in range(C):
                if grid[r, c] != 1:
                    continue
                x0 = ox + c*cell; y0 = oy + r*cell; x1 = x0 + cell; y1 = y0 + cell
                if wall(r, c-1): segs.append((x0, y0, x0, y1))
                if wall(r, c+1): segs.append((x1, y0, x1, y1))
                if wall(r-1, c): segs.append((x0, y0, x1, y0))
                if wall(r+1, c): segs.append((x0, y1, x1, y1))
        w = World(half_width=cell*2.0, style=entry["kind"])
        w.segments = segs
        w.bounds = (ox - cell, ox + C*cell + cell, oy - cell, oy + R*cell + cell)
        pts = {int(k): tuple(v) for k, v in entry["points"].items()}
        return {"world": w, "dock": tuple(entry["dock"]), "points": pts,
                "grid": grid, "cell": cell, "origin": (ox, oy), "name": entry["kind"]}

    def _episode_map(self, opts):
        """Return (map_dict, inflated_planner, plain_planner) for this episode."""
        cm = opts.get("custom_map")
        if cm is not None:        # caller-supplied map dict (e.g. a SLAM-built map)
            return cm, GridPlanner(cm, inflate=1), GridPlanner(cm, inflate=0)
        if self._map_index is not None and not opts.get("map"):
            m = self._load_saved_map(self._map_index[self.rng.integers(len(self._map_index))])
            return m, GridPlanner(m, inflate=1), GridPlanner(m, inflate=0)
        if self.procedural and not opts.get("map"):
            # mostly procedural maps, but mix in the hand-made apartment_complex
            # (with its narrow dock exit + deep niches) so the policy also learns
            # that out-of-distribution geometry.
            if self.rng.random() < self.fixed_mix:
                from apartment_complex_map import build as build_ac
                m = build_ac()
                return m, GridPlanner(m, inflate=1), GridPlanner(m, inflate=0)
            from procedural_delivery import build_procedural
            m = build_procedural(self.rng)
            return m, GridPlanner(m, inflate=1), GridPlanner(m, inflate=0)
        name = opts.get("map") or self.rng.choice(self.map_names)
        if self.procedural:   # named map requested while in procedural mode
            m = build_map(name)
            return m, GridPlanner(m, inflate=1), GridPlanner(m, inflate=0)
        return self.maps[name], self.planners_inf[name], self.planners[name]

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        opts = options or {}
        # domain randomization for this episode
        if self.dr:
            self._lidar_noise = float(self.rng.uniform(0, self.lidar_noise_max))
            self._speed_mult = float(self.rng.uniform(1 - self.speed_jitter, 1 + self.speed_jitter))
            self._turn_mult = float(self.rng.uniform(1 - self.turn_jitter, 1 + self.turn_jitter))
        else:
            self._lidar_noise = 0.0; self._speed_mult = 1.0; self._turn_mult = 1.0

        def route(pinf, pplain, a, b):
            """Plan a->b: inflated route first (centre-hugging), plain as fallback."""
            wps, L = pinf.plan(a, b)
            if wps is None:
                wps, L = pplain.plan(a, b)
            return (wps, L) if (wps is not None and L >= 0.5) else (None, None)

        def build_legs(m, pinf, pplain, ids):
            """Optimal visit order (TSP over real path lengths) + per-leg routes,
            using the inf+plain fallback consistently. Returns (order, legs) or
            (None, None) if some requested point genuinely can't be routed."""
            import itertools
            pts = m["points"]; dock = m["dock"]
            nodes = ["dock"] + list(ids)
            xy = lambda n: dock if n == "dock" else pts[n]
            # pairwise distances with fallback; bail if any required pair unroutable
            dist = {}
            for i in range(len(nodes)):
                for j in range(len(nodes)):
                    if i == j:
                        continue
                    _, L = route(pinf, pplain, xy(nodes[i]), xy(nodes[j]))
                    dist[(i, j)] = L
            def tour_len(perm):
                seq = [0] + [nodes.index(p) for p in perm] + [0]
                tot = 0.0
                for a, b in zip(seq[:-1], seq[1:]):
                    L = dist.get((a, b))
                    if L is None:
                        return None
                    tot += L
                return tot
            best, best_len = None, float("inf")
            for perm in itertools.permutations(ids):
                L = tour_len(perm)
                if L is not None and L < best_len:
                    best_len, best = L, list(perm)
            if best is None:
                return None, None
            legs = []; prev = dock
            for s in best:
                wps, _ = route(pinf, pplain, prev, pts[s]); legs.append(wps); prev = pts[s]
            wps, _ = route(pinf, pplain, prev, dock); legs.append(wps)
            return best, legs

        m = None; legs = None; order = None
        forced_unroutable = False
        for _try in range(16):
            m, pinf, pplain = self._episode_map(opts)
            ids = opts.get("points")
            forced = ids is not None
            if ids is None:
                k = int(self.rng.integers(self.min_parcels, self.max_parcels + 1))
                ids = list(self.rng.choice(list(m["points"]), size=min(k, len(m["points"])),
                                           replace=False))
            order, legs = build_legs(m, pinf, pplain, list(ids))
            if legs is not None:
                break
            if forced:
                # honour the user's choice: don't silently swap points -> report it
                forced_unroutable = True
                break
        if legs is None:
            # could not route the requested points (or no random set worked):
            # fall back to the single nearest point so the env still yields a state
            p0 = min(m["points"], key=lambda p: (m["points"][p][0]-m["dock"][0])**2
                     + (m["points"][p][1]-m["dock"][1])**2)
            order, legs = build_legs(m, pinf, pplain, [p0])
            if legs is None:
                w0, _ = route(pinf, pplain, m["dock"], m["points"][p0]) or ([m["dock"], m["points"][p0]], 0)
                legs = [w0 or [m["dock"], m["points"][p0]], [m["points"][p0], m["dock"]]]
                order = [p0]
        self._reset_vars()
        self.forced_unroutable = forced_unroutable
        self.world = m["world"]; self.map = m; self.order = order or []
        self.legs = legs; self.stops_total = len(legs)
        self.leg_i = 0; self.path = list(self.legs[0]); self.seg_i = 0
        self.pos = np.array(m["dock"], dtype=np.float64)
        d = np.array(self.path[1]) - np.array(self.path[0]) if len(self.path) > 1 else np.array([1., 0.])
        self.heading = math.atan2(d[1], d[0])
        self.prev_along = 0.0
        # spawn dynamic pedestrians + clear the LiDAR stack history
        if self.n_ped > 0:
            from pedestrians2d import Pedestrians
            # pass the map's free-space grid so people stay inside the corridors
            gm = {"grid": m.get("grid"), "cell": m.get("cell"),
                  "origin": m.get("origin")} if m.get("grid") is not None else None
            self.peds = Pedestrians(self.world, self.rng, n=self.n_ped,
                                    speed_range=self.ped_speed, dt=self.dt, grid_map=gm)
        else:
            self.peds = None
        self._lidar_hist = deque(maxlen=self.lidar_stack)
        return self._obs(), {"map": m["name"], "order": list(self.order),
                             "forced_unroutable": forced_unroutable}

    # ---- pure-pursuit + projection ----------------------------------- #
    def _leg_goal(self):
        return np.array(self.path[-1], dtype=np.float64)

    def _project_along(self):
        p = self.pos; best_d = 1e9; acc = 0.0; best_along = 0.0
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

    def _front_clearance(self, half_deg=40.0):
        """Min wall distance (m) in a forward cone -> wall-clearance shaping."""
        n = 9
        offs = np.linspace(-math.radians(half_deg), math.radians(half_deg), n)
        ang = self.heading + offs
        r = self.world.raycast_batch(tuple(self.pos), ang, self.lidar_range)
        return float(np.min(r))

    def _lidar(self):
        ang = self.heading + self.lidar_angles
        r = self.world.raycast_batch(tuple(self.pos), ang, self.lidar_range)
        if self.peds is not None:                       # merge moving people into the scan
            r = self.peds.raycast_into(tuple(self.pos), ang, r, self.lidar_range)
        r = r / self.lidar_range
        if self._lidar_noise > 0:
            r = r + self.rng.normal(0, self._lidar_noise, size=r.shape)
        return np.clip(r, 0.0, 1.0).astype(np.float32)

    def _obs(self, lidar=None):
        lidar = self._lidar() if lidar is None else lidar
        # LiDAR frame-stack: push newest, pad with copies on reset so the policy
        # can read motion (newest fan + older fans -> implicit obstacle velocity).
        if not self._lidar_hist:
            for _ in range(self.lidar_stack):
                self._lidar_hist.append(lidar)
        else:
            self._lidar_hist.append(lidar)
        stacked = np.concatenate(list(self._lidar_hist)) if self.lidar_stack > 1 else lidar

        look = self._lookahead_point()
        d = look - self.pos
        bearing = (math.atan2(d[1], d[0]) - self.heading + math.pi) % (2 * math.pi) - math.pi
        gdist = float(np.linalg.norm(self._leg_goal() - self.pos))
        _, xtrack = self._project_along()
        stops_left = (self.stops_total - self.stops_done) / max(self.stops_total, 1)
        obs = np.concatenate([
            stacked,
            [math.sin(bearing), math.cos(bearing), min(gdist / 20.0, 1.0)],
            [float(np.clip(xtrack / 1.0, 0, 1))],
            [self.prev_turn / self.max_turn],
            [lidar[0]],
            [1.0 if self.grace > 0 else 0.0],
            [stops_left],
        ]).astype(np.float32)
        return np.clip(obs, -1.0, 1.0)

    # ------------------------------------------------------------------ #
    def step(self, action):
        self.step_i += 1
        if self.peds is not None:        # advance moving people first
            self.peds.step(self.dt)
        a = np.clip(np.asarray(action, np.float32), -1, 1)
        # forward maps to [-reverse_frac*max, +max] -> reverse allowed
        f = a[0]
        fwd = (f * (0.5 + 0.5 * self.reverse_frac) + (0.5 - 0.5 * self.reverse_frac)) * self.max_speed
        fwd = float(np.clip(fwd, -self.reverse_frac * self.max_speed, self.max_speed))
        turn = a[1] * self.max_turn
        dturn = abs(turn - self.prev_turn)
        # domain randomization: actuation gain jitter (wheel slip / calibration)
        fwd *= self._speed_mult
        turn_applied = turn * self._turn_mult

        nh = (self.heading + turn_applied * self.dt + math.pi) % (2 * math.pi) - math.pi
        npos = self.pos + np.array([math.cos(nh), math.sin(nh)]) * fwd * self.dt
        wall_hit = self.world.segment_hits_circle(tuple(npos), self.robot_radius)
        ped_hit = (self.peds is not None) and self.peds.hits_robot(tuple(npos), self.robot_radius)
        collided = wall_hit or ped_hit
        if not collided:
            self.pos = npos; self.bump = 0
        else:
            self.bump += 1
        self.heading = nh

        along, xtrack = self._project_along()
        reward = self.w_progress * (along - self.prev_along)
        self.prev_along = along
        reward -= self.w_time
        # heading error to the lookahead point (how sharp the upcoming bend is)
        _look = self._lookahead_point()
        _bearing = (math.atan2(_look[1]-self.pos[1], _look[0]-self.pos[0])
                    - self.heading + math.pi) % (2*math.pi) - math.pi
        sharp_turn = abs(_bearing) > math.radians(20.0)
        # grace: suppress shaping penalties right after a stop switch
        if self.grace > 0:
            self.grace -= 1
        else:
            reward -= self.w_turn * abs(a[1])
            # Smoothness penalty DOUBLED -> hold the wheel steady through S-bends.
            reward -= (2.0 * self.w_dturn) * dturn
            # Dynamic cross-track grace: in a sharp bend (|theta_err|>20deg) relax
            # the cross-track penalty (x0.3) so the robot may cut the inside of the
            # corner instead of being forced onto the exact centre-line (which
            # caused the S-curve weaving). On straights, full penalty.
            xtrack_w = self.w_xtrack * (0.3 if sharp_turn else 1.0)
            reward -= xtrack_w * min(xtrack, 1.0)
            # wall-clearance: penalise hugging a wall (front cone too close)
            front = self._front_clearance()
            if front < self.safe_dist:
                reward -= self.w_near_wall * (self.safe_dist - front) / self.safe_dist
            # straight & stable bonus: moving forward, low steering + low steering-
            # change, with clear space ahead -> reward "đi thẳng ổn định".
            elif fwd > 0 and abs(a[1]) < 0.2 and dturn < 0.2 and not sharp_turn:
                reward += self.w_straight

        # stuck-penalty: if the whole recent window stayed within a small radius,
        # the robot is jammed / spinning -> penalise to push it to explore wider.
        self._pos_hist.append((float(self.pos[0]), float(self.pos[1])))
        if self.grace == 0 and len(self._pos_hist) >= self._pos_hist.maxlen:
            hx = [p[0] for p in self._pos_hist]; hy = [p[1] for p in self._pos_hist]
            spread = max(max(hx) - min(hx), max(hy) - min(hy))
            if spread < self.stuck_radius:
                reward -= self.w_stuck

        # OPTIONAL: clip the DENSE shaping reward (everything before terminal
        # events) to a bounded range -> giảm outlier do nhiều penalty cộng dồn,
        # giúp return đồng đều hơn giữa các map -> Critic fit tốt hơn
        # (explained_variance cao hơn). Terminal events thêm SAU clip nên vẫn mạnh.
        if self.shaping_clip is not None:
            reward = float(np.clip(reward, -self.shaping_clip, self.shaping_clip))

        terminated = False
        info = {"collision": False, "stop": False, "arrived_dock": False,
                "stops_done": self.stops_done}

        if collided:
            reward -= self.w_collide
            if self.bump > self.collision_grace:
                reward -= self.w_jam
                terminated = True; info["collision"] = True

        # reached current leg's target?
        gdist = float(np.linalg.norm(self._leg_goal() - self.pos))
        if not terminated and gdist <= self.arrive_dist:
            is_last = (self.leg_i >= len(self.legs) - 1)
            self.stops_done += 1
            if is_last:
                reward += self.w_dock
                terminated = True
                info["arrived_dock"] = True
            else:
                reward += self.w_stop
                info["stop"] = True
                # CONTINUOUS GOAL TRANSITION: load next leg, grant grace period
                self.leg_i += 1
                self.path = list(self.legs[self.leg_i])
                self.seg_i = 0
                self.prev_along, _ = self._project_along()
                self.grace = self.grace_steps_on_switch

        truncated = self.step_i >= self.max_steps
        self.prev_turn = turn
        return self._obs(), float(reward), terminated, truncated, info
