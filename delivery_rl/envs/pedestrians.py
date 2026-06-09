"""Moving residents (curriculum L4).

Residents walk ALONG the corridor (x-axis) at a realistic, capped speed so the
robot actually meets them head-on or catches up from behind. Each resident
exposes its world position + velocity via :meth:`get_states`, which the reactive
safety shield (``envs/safety_shield.py``) uses to decide when to yield, follow,
sidestep or stop-and-beep.

The motion is a bounded sinusoidal pace: x(t) = x0 + amp*sin(w t + phase) with
w = walk_speed/amp, so the peak speed equals ``walk_speed`` (m/s). One resident
can be made near-stationary to create a "blocking" situation.
TODO(L4): swap for a social-force / ORCA model + recorded trajectories.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import numpy as np
import pybullet as p


@dataclass
class PedState:
    pos: np.ndarray   # world (x, y)
    vel: np.ndarray   # world (vx, vy) m/s
    radius: float
    dist: float       # distance to the robot (filled by get_states)


class PedestrianManager:
    def __init__(self, client: int, config: dict, np_random: np.random.Generator):
        self.client = client
        self.np_random = np_random
        self.world_cfg = config["env"]["world"]
        self.dt = 1.0 / float(config["env"]["control_hz"])
        self.radius = 0.25
        self.height = 1.7

        self.body_ids: List[int] = []
        self._x: List[float] = []    # current world x of each resident
        self._y: List[float] = []    # fixed lane (world y)
        self._vx: List[float] = []   # constant world vx (m/s)
        self._xmin = 0.0
        self._xmax = 0.0

    def reset(self, num_pedestrians: int, robot_xy=None, target_xy=None) -> None:
        for b in self.body_ids:
            p.removeBody(b, physicsClientId=self.client)
        self.body_ids = []
        self._x, self._y, self._vx = [], [], []
        if num_pedestrians <= 0:
            return

        L = self.world_cfg["corridor_length"]
        self._xmin, self._xmax = -L / 2 + 1.2, L / 2 - 1.2
        hy = self.world_cfg["corridor_width"] / 2.0
        rx = float(robot_xy[0]) if robot_xy is not None else 0.0
        tx = float(target_xy[0]) if target_xy is not None else rx + 6.0
        lo, hi = (rx, tx) if rx <= tx else (tx, rx)

        span = max(hi - lo, 1.0)
        for i in range(num_pedestrians):
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=self.radius,
                                         height=self.height, physicsClientId=self.client)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=self.radius, length=self.height,
                                      rgbaColor=[0.92, 0.72, 0.20, 1.0], physicsClientId=self.client)
            # Spread residents along the robot->target span; alternate walking
            # direction so the robot meets some head-on and catches others from
            # behind. All are slow walkers (no permanent blocker), so any block is
            # temporary: the robot beeps, waits/steps aside, then continues.
            x = lo + span * (i + 1) / (num_pedestrians + 1)
            x += float(self.np_random.uniform(-0.8, 0.8))
            # Walk speeds are kept below the robot's max (0.8 m/s) so overtaking
            # a same-direction resident is physically possible.
            if i == 0:
                y = float(self.np_random.uniform(-0.3, 0.3))   # near the centre lane
                ws = float(self.np_random.uniform(0.15, 0.30))  # slow shuffler
            else:
                y = float(self.np_random.uniform(-hy + 0.4, hy - 0.4))
                ws = float(self.np_random.uniform(0.35, 0.55))
            vx = -ws if i % 2 == 1 else ws   # odd-index walk toward -x (head-on)
            x = float(np.clip(x, self._xmin, self._xmax))
            body = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col,
                                     baseVisualShapeIndex=vis, basePosition=[x, y, self.height / 2],
                                     physicsClientId=self.client)
            self.body_ids.append(body)
            self._x.append(x)
            self._y.append(y)
            self._vx.append(vx)

    def step(self, t: float) -> None:
        # constant-velocity walking; wrap to the far end when leaving the corridor
        for i, body in enumerate(self.body_ids):
            self._x[i] += self._vx[i] * self.dt
            if self._x[i] > self._xmax:
                self._x[i] = self._xmin
            elif self._x[i] < self._xmin:
                self._x[i] = self._xmax
            p.resetBasePositionAndOrientation(
                body, [self._x[i], self._y[i], self.height / 2], [0, 0, 0, 1],
                physicsClientId=self.client)

    def get_states(self, robot_xy) -> List[PedState]:
        states: List[PedState] = []
        rx, ry = float(robot_xy[0]), float(robot_xy[1])
        for i, body in enumerate(self.body_ids):
            px, py = self._x[i], self._y[i]
            states.append(PedState(pos=np.array([px, py], dtype=np.float32),
                                   vel=np.array([self._vx[i], 0.0], dtype=np.float32),
                                   radius=self.radius,
                                   dist=float(math.hypot(px - rx, py - ry))))
        return states
