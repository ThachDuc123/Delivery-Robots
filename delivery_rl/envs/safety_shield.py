"""Reactive pedestrian-avoidance safety shield.

Runs ON TOP of the trained RL policy at execution time. The RL policy handles
navigation to the target locker; this shield overrides the commanded base
velocity only when a resident is in the way, implementing the rules:

  * Resident approaching head-on, room to pass  -> SIDESTEP into the gap.
  * Resident approaching head-on, no room       -> YIELD: stop and wait.
  * Robot catching up behind a resident         -> SLOW-FOLLOW; overtake by
                                                   sidestepping once a big enough
                                                   lateral gap exists.
  * Resident directly blocking, cannot pass      -> STOP and BEEP.

This is the standard "planner + reactive safety layer" pattern for service
robots. It is config-gated (``env.safety.enable_pedestrian_shield``) so training
behaviour is unchanged. The shield is sensor-grounded in spirit: it consumes
pedestrian position/velocity that a real robot would obtain from a LiDAR-based
people tracker (here provided directly by the simulator).
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

# status codes (also used by the renderer / overlay)
CLEAR = "clear"
SLOW_FOLLOW = "slow_follow"
YIELD_WAIT = "yield_wait"
SIDESTEP = "sidestep"
BLOCKED_BEEP = "blocked_beep"


class SafetyShield:
    def __init__(self, config: dict, robot_radius: float, corridor_half: float):
        s = config["env"].get("safety", {})
        self.enabled = bool(s.get("enable_pedestrian_shield", True))
        self.concern = float(s.get("concern_distance", 2.6))
        self.slow = float(s.get("slow_distance", 1.8))
        self.stop = float(s.get("stop_distance", 0.9))
        self.disable_radius = float(s.get("disable_near_target", 1.2))
        self.path_halfwidth = float(s.get("path_halfwidth", 0.55))
        self.side_clearance = float(s.get("side_clearance", 0.35))
        self.wall_margin = float(s.get("wall_margin", 0.18))
        self.follow_scale = float(s.get("follow_speed_scale", 0.35))
        self.creep = float(s.get("creep_speed", 0.28))
        self.overtake = float(s.get("overtake_speed", 0.95))
        self.side_speed = float(s.get("sidestep_speed", 0.7))
        self.side_gain = float(s.get("sidestep_gain", 2.5))
        self.robot_radius = robot_radius
        self.corridor_half = corridor_half
        self.commit_steps = int(s.get("commit_steps", 25))
        # hysteresis state: committed pass-side (+1/-1) held for a few steps so
        # the robot does not flip-flop and oscillate in front of a resident
        self._commit_side = 0.0
        self._commit_timer = 0

    def reset(self) -> None:
        self._commit_side = 0.0
        self._commit_timer = 0

    # ------------------------------------------------------------------ #
    def filter(self, action, robot_xy, yaw: float, target_xy, ped_states
               ) -> Tuple[np.ndarray, str, bool]:
        """Return (adjusted body-frame action, status, beep)."""
        a = np.asarray(action, dtype=np.float32).copy()
        if not self.enabled or not ped_states:
            self._commit_timer = 0
            self._commit_side = 0.0
            return a, CLEAR, False

        robot_xy = np.asarray(robot_xy, dtype=np.float32)
        to_tgt = np.asarray(target_xy, dtype=np.float32) - robot_xy
        dtt = float(np.linalg.norm(to_tgt))
        if dtt < self.disable_radius:           # let docking finish unhindered
            return a, CLEAR, False
        travel = to_tgt / dtt
        lateral = np.array([-travel[1], travel[0]], dtype=np.float32)

        # nearest concerning resident ahead, inside the path band
        best = None
        for ped in ped_states:
            rel = ped.pos - robot_xy
            along = float(rel @ travel)
            perp = float(rel @ lateral)
            if along < -0.3 or along > self.concern:
                continue
            if abs(perp) > self.path_halfwidth + ped.radius:
                continue
            if best is None or along < best[0]:
                best = (along, perp, ped)
        if best is None:
            self._commit_timer = max(0, self._commit_timer - 1)
            if self._commit_timer == 0:
                self._commit_side = 0.0
            return a, CLEAR, False

        along, perp, ped = best
        ped_along_vel = float(ped.vel @ travel)
        approaching = ped_along_vel < -0.05
        stationary = float(np.linalg.norm(ped.vel)) < 0.12

        # is there room to slip past on the side away from the resident?
        # Hysteresis: once we commit to a side, keep it while still engaged so we
        # do not oscillate left/right in front of the resident.
        if self._commit_timer > 0 and self._commit_side != 0.0:
            free_dir = self._commit_side
        else:
            free_dir = -1.0 if perp >= 0 else 1.0
            self._commit_side = free_dir
        self._commit_timer = self.commit_steps
        robot_lat = float(robot_xy @ lateral)
        ped_lat = float(ped.pos @ lateral)
        # keep a safety margin off the wall so weaving around a person never
        # grazes the corridor wall (a wall hit IS mission-ending)
        usable = self.corridor_half - self.robot_radius - self.wall_margin
        limit = usable * free_dir
        gap = (limit - ped_lat) if free_dir > 0 else (ped_lat - limit)
        need = self.robot_radius + ped.radius + self.side_clearance
        can_side = gap > need
        target_lat = ped_lat + free_dir * need
        target_lat = float(np.clip(target_lat, -usable, usable))
        lat_err = target_lat - robot_lat

        # decide a desired WORLD-frame velocity (fractions of max speed)
        def world_cmd(forward_frac, lateral_frac):
            wv = travel * forward_frac + lateral * lateral_frac
            return wv

        status = CLEAR
        beep = False
        wv = None
        if along <= self.stop:
            if can_side:
                wv = world_cmd(self.creep, np.clip(lat_err * self.side_gain,
                                                   -self.side_speed, self.side_speed))
                status = SIDESTEP
            else:
                wv = np.zeros(2, dtype=np.float32)
                status = BLOCKED_BEEP
                beep = True
        elif along <= self.slow:
            if approaching:
                if can_side:
                    wv = world_cmd(self.creep, np.clip(lat_err * self.side_gain,
                                                       -self.side_speed, self.side_speed))
                    status = SIDESTEP
                else:
                    wv = np.zeros(2, dtype=np.float32)
                    status = YIELD_WAIT
            else:  # same direction -> following behind
                if can_side:
                    # OVERTAKE: move into the side gap AND drive forward faster
                    # than the resident so we can actually pass them.
                    wv = world_cmd(self.overtake, np.clip(lat_err * self.side_gain,
                                                          -self.side_speed, self.side_speed))
                    status = SIDESTEP
                else:
                    wv = world_cmd(self.follow_scale, 0.0)
                    status = SLOW_FOLLOW
        else:
            return a, CLEAR, False

        # convert desired world velocity -> body frame (vx forward, vy left)
        c, s = math.cos(yaw), math.sin(yaw)
        vbx = c * wv[0] + s * wv[1]
        vby = -s * wv[0] + c * wv[1]
        a[0] = float(np.clip(vbx, -1.0, 1.0))
        a[1] = float(np.clip(vby, -1.0, 1.0))
        if status in (YIELD_WAIT, BLOCKED_BEEP):
            a[2] *= 0.15   # stop spinning while waiting
        else:
            a[2] *= 0.5    # damp turning while manoeuvring around people
        return a, status, beep
