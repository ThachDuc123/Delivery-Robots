"""Reactive pedestrian-avoidance safety shield (Stage 3, no extra training).

Runs ON TOP of the trained navigation policy (`ms_mixed`, which already does
curves + multi-stop at ~100%). The RL policy proposes a [forward, turn] action
from its normal (people-free) observation; this shield then inspects the nearby
pedestrians (their position + velocity) and overrides the command per simple,
robust rules:

  * person close & in the robot's forward path  -> SLOW or STOP (yield),
  * person to one side blocking the chosen way   -> STEER toward the freer side,
  * path clear                                   -> pass the action through.

This is the standard "planner/RL + reactive safety layer" used on real delivery
robots, and it avoids the (failed) attempt to make one network learn curves +
multi-stop + dynamic dodging all at once.
"""

from __future__ import annotations

import math
import numpy as np


class PedShield:
    def __init__(self, robot_radius=0.22, ped_radius=0.28,
                 react_dist=1.5, stop_dist=0.7, fov_deg=55.0, side_clear=0.9):
        self.rr = robot_radius
        self.pr = ped_radius
        self.react = react_dist      # start reacting within this range (m)
        self.stop = stop_dist        # hard stop within this range (m)
        self.fov = math.radians(fov_deg)   # half-cone counted as "ahead"
        self.side_clear = side_clear

    def filter_lidar(self, action, lidar, lidar_angles, lidar_range,
                     wall_lidar=None):
        """LiDAR forward-cone shield. Scans a narrow forward cone; if an obstacle
        is closer than `react`, brake / steer to the freer side; if very close,
        stop / reverse; else hand control back to the RL policy.

        To avoid reacting to the *static walls* of a tight corridor (which would
        freeze the robot), pass ``wall_lidar`` = the scan expected from the known
        map walls only. The shield then reacts to a ray only where the live scan
        is markedly SHORTER than the wall-only scan, i.e. a DYNAMIC obstacle
        (person) has appeared in front of the wall -- a simple moving-object
        detector. Without ``wall_lidar`` it reacts to any near return."""
        a = np.array(action, dtype=np.float32).copy()
        rng = np.asarray(lidar, dtype=np.float32) * lidar_range
        ang = np.asarray(lidar_angles, dtype=np.float32)
        aw = (ang + math.pi) % (2 * math.pi) - math.pi
        cone = np.abs(aw) < self.fov
        if not np.any(cone):
            return a, "clear"
        cone_d = rng[cone].copy(); cone_a = aw[cone]
        if wall_lidar is not None:
            wall_d = np.asarray(wall_lidar, dtype=np.float32)[cone] * lidar_range
            # a ray only counts as a (dynamic) obstacle if it's >=0.35m shorter
            # than the bare wall would give -> ignore the static corridor walls
            dynamic = cone_d < (wall_d - 0.35)
            cone_d = np.where(dynamic, cone_d, lidar_range)
        dmin = float(np.min(cone_d))
        if dmin >= self.react:
            return a, "clear"
        # which side is freer? compare mean clearance left vs right of centre
        left_free = float(np.mean(cone_d[cone_a > 0])) if np.any(cone_a > 0) else 0.0
        right_free = float(np.mean(cone_d[cone_a < 0])) if np.any(cone_a < 0) else 0.0
        steer_to_left = left_free > right_free      # turn toward the more open side
        side = 1.0 if steer_to_left else -1.0
        if dmin < self.stop:
            a[0] = -1.0                              # brake / slight reverse
            a[1] = float(np.clip(side * 0.8 + a[1] * 0.2, -1, 1))
            return a, "stop_yield"
        slow = (dmin - self.stop) / (self.react - self.stop)
        a[0] = float(np.clip(min(a[0], slow * 0.4 - 0.2), -1, 1))
        a[1] = float(np.clip(side * 0.6 + a[1] * 0.4, -1, 1))
        return a, "slow_sidestep"

    def filter(self, action, pos, heading, peds):
        """Return (adjusted_action, status). peds.pos/.vel are world arrays."""
        a = np.array(action, dtype=np.float32).copy()
        if peds is None or len(peds.pos) == 0:
            return a, "clear"
        pos = np.array(pos, dtype=np.float64)
        fwd = np.array([math.cos(heading), math.sin(heading)])
        left = np.array([-fwd[1], fwd[0]])

        # consider the most threatening person ahead (closest, predicted a bit
        # forward in time so we react to where they're GOING, not just where they are)
        worst = None
        for i in range(len(peds.pos)):
            rel = peds.pos[i] - pos
            # predict ~0.5s ahead using the person's velocity
            rel_pred = rel + peds.vel[i] * 0.5
            d = float(np.linalg.norm(rel))
            dahead = float(rel @ fwd)
            ang = abs(math.atan2(rel @ left, rel @ fwd))
            in_path = (dahead > 0) and (ang < self.fov) and (d < self.react)
            # also flag a person whose predicted pos lands in front & close
            d_pred = float(np.linalg.norm(rel_pred))
            closing = (rel_pred @ fwd > 0) and (d_pred < self.react) and \
                      (abs(rel_pred @ left) < (self.rr + self.pr + 0.4))
            if in_path or closing:
                if worst is None or d < worst[0]:
                    side = float(rel @ left)   # >0 person on left, <0 on right
                    worst = (d, side)
        if worst is None:
            return a, "clear"

        d, side = worst
        if d < self.stop:
            # too close in front -> stop forward motion (yield), allow a small turn
            a[0] = -1.0     # maps to ~ -reverse..min forward -> near 0/slightly back
            a[1] = float(np.clip(-np.sign(side) * 0.8 + a[1] * 0.2, -1, 1))
            return a, "stop_yield"
        else:
            # in the slow band -> reduce speed and steer to the freer side
            slow = (d - self.stop) / (self.react - self.stop)   # 0..1
            a[0] = float(np.clip(min(a[0], slow * 0.4 - 0.2), -1, 1))   # creep/slow
            a[1] = float(np.clip(-np.sign(side) * 0.6 + a[1] * 0.4, -1, 1))  # veer away
            return a, "slow_sidestep"
