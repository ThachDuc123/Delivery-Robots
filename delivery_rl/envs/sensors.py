"""Non-visual sensor suite: 2D LiDAR, ToF, IMU and wheel odometry.

Everything is computed with PyBullet ray casts and the robot's kinematic state
plus configurable Gaussian noise. NO camera / NO images are produced here.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pybullet as p


class SensorSuite:
    IMU_DIM = 7      # ax, ay, az, wx, wy, wz, yaw
    ODOM_DIM = 6     # x, y, yaw, vx, vy, omega

    def __init__(self, client: int, config: dict, np_random: np.random.Generator):
        self.client = client
        self.np_random = np_random
        s = config["env"]["sensors"]

        self.lidar_n = int(s["lidar"]["num_rays"])
        self.lidar_range = float(s["lidar"]["max_range"])
        self.lidar_h = float(s["lidar"]["height"])
        self.lidar_noise = float(s["lidar"]["noise_std"])
        self.lidar_angles = np.linspace(0, 2 * math.pi, self.lidar_n, endpoint=False)

        self.tof_range = float(s["tof"]["max_range"])
        self.tof_noise = float(s["tof"]["noise_std"])
        self.tof_angles = np.array([0.0, math.pi / 2, math.pi, -math.pi / 2])  # F,L,B,R

        imu = s["imu"]
        self.accel_noise = float(imu["accel_noise_std"])
        self.gyro_noise = float(imu["gyro_noise_std"])
        self.imu_yaw_noise = float(imu["yaw_noise_std"])

        od = s["odometry"]
        self.odom_xy_noise = float(od["xy_noise_std"])
        self.odom_yaw_noise = float(od["yaw_noise_std"])
        self.odom_vel_noise = float(od["vel_noise_std"])

        self.noise_scale = 1.0  # raised by domain randomization (L5)

    @property
    def lidar_dim(self) -> int:
        return self.lidar_n

    @property
    def tof_dim(self) -> int:
        return 4

    @property
    def total_dim(self) -> int:
        return self.lidar_dim + self.tof_dim + self.IMU_DIM + self.ODOM_DIM

    # ------------------------------------------------------------------ #
    def _raycast_normalized(self, pos, yaw, angles, max_range) -> np.ndarray:
        origin = [pos[0], pos[1], self.lidar_h]
        ray_from, ray_to = [], []
        for a in angles:
            d = yaw + a
            ray_from.append(origin)
            ray_to.append([origin[0] + math.cos(d) * max_range,
                           origin[1] + math.sin(d) * max_range, origin[2]])
        results = p.rayTestBatch(ray_from, ray_to, physicsClientId=self.client)
        fractions = np.array([r[2] for r in results], dtype=np.float32)  # 1.0 == no hit
        noise = self.np_random.normal(0, self.lidar_noise * self.noise_scale, size=fractions.shape)
        return np.clip(fractions + noise, 0.0, 1.0).astype(np.float32)

    def read_lidar(self, pos, yaw) -> np.ndarray:
        return self._raycast_normalized(pos, yaw, self.lidar_angles, self.lidar_range)

    def read_tof(self, pos, yaw) -> np.ndarray:
        return self._raycast_normalized(pos, yaw, self.tof_angles, self.tof_range)

    def read_imu(self, accel_body, omega, yaw) -> np.ndarray:
        ax, ay = accel_body
        vec = np.array([ax, ay, 0.0, 0.0, 0.0, omega, self._wrap(yaw)], dtype=np.float32)
        scale = np.array([self.accel_noise, self.accel_noise, self.accel_noise,
                          self.gyro_noise, self.gyro_noise, self.gyro_noise,
                          self.imu_yaw_noise], dtype=np.float32) * self.noise_scale
        vec = vec + self.np_random.normal(0, 1, size=vec.shape).astype(np.float32) * scale
        vec[6] = self._wrap(vec[6])
        return vec

    def read_odometry(self, x, y, yaw, vx, vy, omega) -> np.ndarray:
        vec = np.array([x, y, self._wrap(yaw), vx, vy, omega], dtype=np.float32)
        scale = np.array([self.odom_xy_noise, self.odom_xy_noise, self.odom_yaw_noise,
                          self.odom_vel_noise, self.odom_vel_noise, self.odom_vel_noise],
                         dtype=np.float32) * self.noise_scale
        vec = vec + self.np_random.normal(0, 1, size=vec.shape).astype(np.float32) * scale
        vec[2] = self._wrap(vec[2])
        return vec

    @staticmethod
    def _wrap(angle: float) -> float:
        return (angle + math.pi) % (2 * math.pi) - math.pi
