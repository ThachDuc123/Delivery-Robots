"""Mecanum delivery robot.

Loads a URDF when one is provided; otherwise builds a PyBullet primitive robot
(base box + 4 visual wheels) so the project runs with no assets.

The base is driven HOLONOMICALLY (kinematic integration of vx/vy/omega) and
collisions are resolved by querying PyBullet contacts and refusing penetrating
moves.  This is the deliberate fallback for the sensor-only navigation phase;
the mecanum inverse kinematics (`mecanum_wheel_speeds`) is still computed and
exposed for the energy term and for future real wheel-torque control.

TODO(asset): when ``robot.urdf`` (with mecanum rollers) exists, replace the
holonomic shortcut in `apply_velocity` with wheel-velocity motor control.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Tuple

import numpy as np
import pybullet as p


class MecanumRobot:
    def __init__(self, client: int, config: dict, np_random: np.random.Generator,
                 base_dir: str):
        self.client = client
        self.np_random = np_random
        self.base_dir = base_dir

        rcfg = config["env"]["robot"]
        mcfg = config["env"]["mechanism"]
        self.urdf_path = os.path.join(base_dir, rcfg["urdf"])
        self.base_size = list(rcfg["base_size"])
        self.base_mass = float(rcfg["base_mass"])
        self.wheel_radius = float(rcfg["wheel_radius"])
        self.lx = float(rcfg["half_wheel_base"])
        self.ly = float(rcfg["half_track_width"])
        self.spawn_height = float(rcfg["spawn_height"])
        self.max_linear_speed = float(rcfg["max_linear_speed"])
        self.max_yaw_rate = float(rcfg["max_yaw_rate"])

        self.arm_range = mcfg["arm_lift_range"]
        self.tray_range = mcfg["tray_extend_range"]
        self.probe_range = mcfg["probe_auth_range"]
        self.carousel_slots = int(mcfg["carousel_slots"])
        self.macro_steps = int(mcfg["macro_steps"])

        self.dt = 1.0 / float(config["env"]["control_hz"])
        self.battery_capacity = float(config["env"]["task"]["battery_capacity"])
        self.drain_per_m = float(config["env"]["task"]["battery_drain_per_m"])
        self.drain_idle = float(config["env"]["task"]["battery_drain_idle"])

        self.body_id: int = -1
        self.x = self.y = self.yaw = 0.0
        self.world_v = np.zeros(2, dtype=np.float32)
        self.prev_world_v = np.zeros(2, dtype=np.float32)
        self.omega = 0.0
        self.prev_omega = 0.0
        self.accel_body = np.zeros(2, dtype=np.float32)
        self.tilt_deg = 0.0
        self.bumper = 0.0
        self.battery = self.battery_capacity

        self.arm_lift = 0.0
        self.tray_extend = 0.0
        self.probe_auth = 0.0
        self.carousel_index = 0
        self._macro_t = -1   # -1 == inactive

    # ------------------------------------------------------------------ #
    def load(self) -> None:
        if os.path.isfile(self.urdf_path):
            # TODO(asset): URDF should expose 4 mecanum wheel joints + arm joints.
            self.body_id = p.loadURDF(self.urdf_path, [0, 0, self.spawn_height],
                                      physicsClientId=self.client)
        else:
            self.body_id = self._build_primitive()

    def _build_primitive(self) -> int:
        hx, hy, hz = [s / 2 for s in self.base_size]
        base_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz],
                                          physicsClientId=self.client)
        base_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, hz],
                                       rgbaColor=[0.2, 0.5, 0.95, 1.0],
                                       physicsClientId=self.client)
        wheel_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=self.wheel_radius,
                                        length=0.04, rgbaColor=[0.1, 0.1, 0.12, 1.0],
                                        physicsClientId=self.client)
        wheel_orn = p.getQuaternionFromEuler([math.pi / 2, 0, 0])
        offs = [(hx * 0.8, hy, -hz), (hx * 0.8, -hy, -hz),
                (-hx * 0.8, hy, -hz), (-hx * 0.8, -hy, -hz)]
        n = 4
        return p.createMultiBody(
            baseMass=self.base_mass,
            baseCollisionShapeIndex=base_col,
            baseVisualShapeIndex=base_vis,
            basePosition=[0, 0, self.spawn_height],
            linkMasses=[0.0] * n,
            linkCollisionShapeIndices=[-1] * n,
            linkVisualShapeIndices=[wheel_vis] * n,
            linkPositions=list(offs),
            linkOrientations=[wheel_orn] * n,
            linkInertialFramePositions=[[0, 0, 0]] * n,
            linkInertialFrameOrientations=[[0, 0, 0, 1]] * n,
            linkParentIndices=[0] * n,
            linkJointTypes=[p.JOINT_FIXED] * n,
            linkJointAxis=[[0, 0, 1]] * n,
            physicsClientId=self.client,
        )

    # ------------------------------------------------------------------ #
    def reset(self, x: float, y: float, yaw: float) -> None:
        self.x, self.y, self.yaw = float(x), float(y), float(yaw)
        self.world_v[:] = 0.0
        self.prev_world_v[:] = 0.0
        self.omega = self.prev_omega = 0.0
        self.accel_body[:] = 0.0
        self.tilt_deg = 0.0
        self.bumper = 0.0
        self.battery = self.battery_capacity
        self.arm_lift = self.tray_extend = self.probe_auth = 0.0
        self.carousel_index = 0
        self._macro_t = -1
        self._teleport()

    def _teleport(self) -> None:
        orn = p.getQuaternionFromEuler([0, 0, self.yaw])
        p.resetBasePositionAndOrientation(self.body_id, [self.x, self.y, self.spawn_height],
                                          orn, physicsClientId=self.client)

    # ------------------------------------------------------------------ #
    def mecanum_wheel_speeds(self, vx: float, vy: float, omega: float) -> np.ndarray:
        """Inverse kinematics -> [FL, FR, RL, RR] wheel angular speeds (rad/s)."""
        r = self.wheel_radius
        k = self.lx + self.ly
        return np.array([
            (vx - vy - k * omega) / r,
            (vx + vy + k * omega) / r,
            (vx + vy - k * omega) / r,
            (vx - vy + k * omega) / r,
        ], dtype=np.float32)

    def apply_velocity(self, vx: float, vy: float, omega: float,
                       collision_ids: List[int], pedestrian_ids: List[int],
                       floor_id: int) -> Dict:
        """vx, vy in m/s (body frame), omega in rad/s -- already scaled."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        wvx = vx * c - vy * s
        wvy = vx * s + vy * c
        nx = self.x + wvx * self.dt
        ny = self.y + wvy * self.dt
        nyaw = self._wrap(self.yaw + omega * self.dt)

        prev_pose = (self.x, self.y, self.yaw)
        self.x, self.y, self.yaw = nx, ny, nyaw
        self._teleport()

        hit_world, hit_ped, max_pen = self._detect_collision(pedestrian_ids, floor_id)
        if hit_world or hit_ped:
            self.x, self.y, self.yaw = prev_pose
            self._teleport()
            moved = 0.0
            self.world_v[:] = 0.0
            self.omega = 0.0
        else:
            moved = math.hypot(wvx, wvy) * self.dt
            self.world_v[:] = (wvx, wvy)
            self.omega = omega

        aw = (self.world_v - self.prev_world_v) / self.dt
        self.accel_body[0] = c * aw[0] + s * aw[1]
        self.accel_body[1] = -s * aw[0] + c * aw[1]
        self.prev_world_v[:] = self.world_v
        d_omega = self.omega - self.prev_omega
        self.prev_omega = self.omega

        self.battery = max(0.0, self.battery - moved * self.drain_per_m - self.drain_idle)
        self.bumper = 1.0 if (hit_world or hit_ped) else 0.0
        _, orn = p.getBasePositionAndOrientation(self.body_id, physicsClientId=self.client)
        roll, pitch, _ = p.getEulerFromQuaternion(orn)
        self.tilt_deg = math.degrees(max(abs(roll), abs(pitch)))

        return {
            "collision_world": hit_world,
            "collision_pedestrian": hit_ped,
            "penetration": max_pen,
            "distance_moved": moved,
            "d_omega": abs(d_omega),
            "wheel_speeds": self.mecanum_wheel_speeds(vx, vy, omega),
            "tilt_deg": self.tilt_deg,
        }

    def _detect_collision(self, pedestrian_ids, floor_id) -> Tuple[bool, bool, float]:
        p.performCollisionDetection(physicsClientId=self.client)
        contacts = p.getContactPoints(bodyA=self.body_id, physicsClientId=self.client)
        hit_world = hit_ped = False
        max_pen = 0.0
        ped = set(pedestrian_ids)
        for ct in contacts:
            other = ct[2]
            dist = ct[8]  # contactDistance, < 0 == penetration
            if other == floor_id or other == self.body_id:
                continue
            if dist < 0.002:
                if other in ped:
                    hit_ped = True
                else:
                    hit_world = True
                max_pen = max(max_pen, -dist)
        return hit_world, hit_ped, max_pen

    # ------------------------------------------------------------------ #
    #  Delivery macro (lift -> extend -> auth -> release -> retract)
    # ------------------------------------------------------------------ #
    def start_macro(self) -> None:
        self._macro_t = 0

    @property
    def macro_active(self) -> bool:
        return self._macro_t >= 0

    def step_macro(self) -> bool:
        """Advance the parametrised macro one control step. True when finished."""
        if self._macro_t < 0:
            return False
        t = self._macro_t / max(self.macro_steps - 1, 1)
        arm_hi, tray_hi, probe_hi = self.arm_range[1], self.tray_range[1], self.probe_range[1]
        if t < 0.25:
            self.arm_lift = arm_hi * (t / 0.25)
        elif t < 0.50:
            self.arm_lift = arm_hi
            self.tray_extend = tray_hi * ((t - 0.25) / 0.25)
        elif t < 0.65:
            self.tray_extend = tray_hi
            self.probe_auth = probe_hi
        elif t < 0.80:
            self.probe_auth = 0.0
        else:
            k = (t - 0.80) / 0.20
            self.tray_extend = tray_hi * (1 - k)
            self.arm_lift = arm_hi * (1 - k)
        self._macro_t += 1
        if self._macro_t >= self.macro_steps:
            self._macro_t = -1
            self.arm_lift = self.tray_extend = self.probe_auth = 0.0
            return True
        return False

    def advance_carousel(self) -> None:
        self.carousel_index = (self.carousel_index + 1) % self.carousel_slots

    # ------------------------------------------------------------------ #
    def mechanism_obs(self) -> np.ndarray:
        return np.array([
            self.arm_lift / max(self.arm_range[1], 1e-6),
            self.tray_extend / max(self.tray_range[1], 1e-6),
            self.carousel_index / max(self.carousel_slots - 1, 1),
            self.bumper,
            self.battery / max(self.battery_capacity, 1e-6),
        ], dtype=np.float32)

    def get_pose(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.yaw

    def body_velocity(self) -> Tuple[float, float, float]:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        vbx = c * self.world_v[0] + s * self.world_v[1]
        vby = -s * self.world_v[0] + c * self.world_v[1]
        return float(vbx), float(vby), float(self.omega)

    @staticmethod
    def _wrap(angle: float) -> float:
        return (angle + math.pi) % (2 * math.pi) - math.pi
