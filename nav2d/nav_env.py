"""2D sensor-only corridor navigation (Gymnasium, numpy only).

The robot must reach a GOAL it only knows as a *relative bearing + distance*
(like "the parcel locker is 12 m that way"), then RETURN to its start -- using
only a LiDAR fan + its own odometry. There is **no global planner and no map**:
the policy has to follow whatever corridor it is dropped into (straight, arc,
L/U/S-turns, niches), so a trained policy generalises to unseen layouts.

Observation (flat, all ~[-1,1], NO vision):
    [ n_lidar normalized ranges
    | goal: sin(bearing), cos(bearing), dist_norm
    | heading-progress helpers: forward-clearance, prev_action(2)
    | phase flag (0 = outbound, 1 = returning) ]
Action: Box(2,) = [forward_speed in [0,1]-ish (-1..1 mapped), turn_rate in [-1,1]]
    A differential-drive style command -> simple, robust to learn in 2D.

Reward: dense progress toward the current target + arrival bonus + smoothness;
penalties for hitting walls, spinning, and time. Episode ends on collision,
timeout, or successful round-trip (reach goal then return to start).
"""

from __future__ import annotations

import math
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from world2d import STYLES, World


class Nav2DEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, config: Optional[dict] = None, render_mode: Optional[str] = None):
        super().__init__()
        c = config or {}
        self.render_mode = render_mode

        # robot
        self.robot_radius = 0.22
        self.max_speed = float(c.get("max_speed", 0.9))        # m/s
        self.max_turn = float(c.get("max_turn", 2.2))          # rad/s
        self.dt = float(c.get("dt", 0.1))                      # 10 Hz control

        # lidar
        self.n_lidar = int(c.get("n_lidar", 24))
        self.lidar_range = float(c.get("lidar_range", 5.0))
        self.lidar_fov = float(c.get("lidar_fov", 2 * math.pi))  # full 360
        self.lidar_noise = float(c.get("lidar_noise", 0.0))
        if self.lidar_fov >= 2 * math.pi - 1e-6:
            self.lidar_angles = np.linspace(0, 2 * math.pi, self.n_lidar, endpoint=False)
        else:
            self.lidar_angles = np.linspace(-self.lidar_fov / 2, self.lidar_fov / 2, self.n_lidar)

        # task
        self.max_steps = int(c.get("max_steps", 800))
        self.reach_dist = float(c.get("reach_dist", 0.6))
        self.round_trip = bool(c.get("round_trip", True))
        # "world_kind": "simple" (single corridor) | "hard" (multi-corridor
        # junctions/grids/loops) | "mixed" (both pools)
        self.world_kind = c.get("world_kind", "simple")
        self.styles = c.get("styles", None)   # explicit style list overrides kind
        self.fixed_style = c.get("style", None)

        # reward weights
        self.w_progress = float(c.get("w_progress", 1.2))
        self.w_goal = float(c.get("w_goal", 30.0))
        self.w_return = float(c.get("w_return", 30.0))
        self.w_collide = float(c.get("w_collide", 12.0))
        self.w_time = float(c.get("w_time", 0.01))
        self.w_spin = float(c.get("w_spin", 0.02))
        self.w_clear = float(c.get("w_clear", 0.05))      # penalty weight for wall-hugging
        self.clear_thresh = float(c.get("clear_thresh", 0.18))  # lidar-frac below = "too close"
        self.w_bump = float(c.get("w_bump", 0.3))         # small per-step bump penalty
        self.collision_grace = int(c.get("collision_grace", 25))  # consecutive stuck steps -> terminate

        obs_dim = self.n_lidar + 3 + 1 + 2 + 1
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self.rng = np.random.default_rng(0)
        self.world: Optional[World] = None
        self._reset_state()

    def _reset_state(self):
        self.pos = np.zeros(2, dtype=np.float64)
        self.heading = 0.0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.step_i = 0
        self.phase = 0            # 0 outbound (to goal), 1 returning (to start)
        self.prev_dist = 0.0
        self.start_pt = np.zeros(2)
        self.goal_pt = np.zeros(2)
        self.trail = []
        self._bump_count = 0

    # ------------------------------------------------------------------ #
    def _make_world(self, style):
        """Build a world honouring world_kind / explicit style.

        A style name belonging to the hard pool always uses the hard generator,
        regardless of world_kind, so eval can request any specific layout."""
        from world_hard import HARD_STYLES, generate_hard
        if style is not None:
            if style in HARD_STYLES:
                return generate_hard(self.rng, style=style)
            return World.generate(self.rng, style=style)
        kind = self.world_kind
        if kind == "hard":
            return generate_hard(self.rng)
        if kind == "mixed":
            if self.rng.random() < 0.5:
                return generate_hard(self.rng)
            return World.generate(self.rng)
        return World.generate(self.rng)

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        style = (options or {}).get("style", self.fixed_style)
        self.world = self._make_world(style)
        self._reset_state()

        self.pos = np.array(self.world.start, dtype=np.float64)
        self.heading = self.world.start_heading
        self.start_pt = np.array(self.world.start, dtype=np.float64)
        self.goal_pt = np.array(self.world.goal, dtype=np.float64)
        # nudge inside so we never spawn on a wall
        self.prev_dist = float(np.linalg.norm(self.goal_pt - self.pos))
        self.trail = [tuple(self.pos)]
        return self._obs(), {"style": self.world.style}

    # ------------------------------------------------------------------ #
    def _current_target(self):
        return self.goal_pt if self.phase == 0 else self.start_pt

    def _lidar(self):
        rngs = np.empty(self.n_lidar, dtype=np.float32)
        for i, a in enumerate(self.lidar_angles):
            d = self.world.raycast(tuple(self.pos), self.heading + a, self.lidar_range)
            rngs[i] = d
        if self.lidar_noise > 0:
            rngs = rngs + self.rng.normal(0, self.lidar_noise, size=rngs.shape)
        return np.clip(rngs / self.lidar_range, 0.0, 1.0)

    def _obs(self, lidar=None):
        lidar = self._lidar() if lidar is None else lidar
        tgt = self._current_target()
        d = tgt - self.pos
        dist = float(np.linalg.norm(d))
        bearing = math.atan2(d[1], d[0]) - self.heading
        bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
        diag = float(np.hypot(self.world.bounds[1] - self.world.bounds[0],
                              self.world.bounds[3] - self.world.bounds[2])) or 1.0
        fwd_clear = lidar[0]  # ray straight ahead (angle 0 is index 0)
        obs = np.concatenate([
            lidar,
            [math.sin(bearing), math.cos(bearing), min(dist / diag, 1.0)],
            [fwd_clear],
            self.prev_action,
            [float(self.phase)],
        ]).astype(np.float32)
        return np.clip(obs, -1.0, 1.0)

    # ------------------------------------------------------------------ #
    def step(self, action):
        self.step_i += 1
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        fwd = (a[0] * 0.5 + 0.5) * self.max_speed     # map [-1,1] -> [0, max] (no reverse)
        turn = a[1] * self.max_turn

        self.heading = (self.heading + turn * self.dt + math.pi) % (2 * math.pi) - math.pi
        step_vec = np.array([math.cos(self.heading), math.sin(self.heading)]) * fwd * self.dt
        new_pos = self.pos + step_vec

        collided = self.world.segment_hits_circle(tuple(new_pos), self.robot_radius)
        if not collided:
            self.pos = new_pos
            self.trail.append(tuple(self.pos))
            self._bump_count = 0
        else:
            # blocked: stay put (policy can steer away next step).
            self._bump_count += 1

        # reward
        tgt = self._current_target()
        dist = float(np.linalg.norm(tgt - self.pos))
        reward = self.w_progress * (self.prev_dist - dist)
        self.prev_dist = dist
        reward -= self.w_time
        reward -= self.w_spin * abs(a[1])

        # clearance shaping (computed ONCE here, reused for the obs below): keep a
        # safe margin from walls so tight branch corridors are navigated without
        # scraping. This is the real fix for the collision-dominated branch case.
        lidar_now = self._lidar()
        min_clear = float(np.min(lidar_now))
        if min_clear < self.clear_thresh:
            reward -= self.w_clear * (self.clear_thresh - min_clear) / self.clear_thresh

        terminated = False
        info = {"style": self.world.style, "phase": self.phase, "collision": False,
                "reached_goal": False, "round_trip": False}

        if collided:
            # Light recoverable bump: small penalty, robot stays put and may steer
            # out (lets it back out of a dead-end branch). Only a *persistent* jam
            # (stuck many steps in a row) ends the episode -- this avoids the heavy
            # repeated -w_collide noise that previously wrecked the return trip.
            reward -= self.w_bump
            if self._bump_count > self.collision_grace:
                reward -= self.w_collide
                terminated = True
                info["collision"] = True
        elif dist <= self.reach_dist:
            if self.phase == 0:
                reward += self.w_goal
                info["reached_goal"] = True
                if self.round_trip:
                    self.phase = 1
                    self.prev_dist = float(np.linalg.norm(self.start_pt - self.pos))
                else:
                    terminated = True
                    info["round_trip"] = True
            else:
                reward += self.w_return
                terminated = True
                info["round_trip"] = True

        truncated = self.step_i >= self.max_steps
        self.prev_action = a.copy()
        obs = self._obs(lidar=lidar_now)   # reuse this step's scan (no double raycast)
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ #
    def render(self):
        return None  # rendering handled by render2d.py to keep the env light
