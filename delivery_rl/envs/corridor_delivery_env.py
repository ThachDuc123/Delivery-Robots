"""Gymnasium environment: sensor-only delivery robot in an apartment corridor.

Observation (flat Box, all in [-1, 1], NO camera / NO vision):
    [ 36 LiDAR | 4 ToF | 7 IMU | 6 odometry | 3 rel-pose-to-target
      | num_lockers remaining-mask | 1 parcels-carried | 5 mechanism ]

Action (Box(3,) in [-1, 1]):
    [vx, vy, omega] mecanum base velocities (scaled by the configured maxima).
    The delivery macro auto-triggers in a locker dock zone (hook to learn later).
"""

from __future__ import annotations

import math
import os
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

from delivery_rl.configs.loader import default_config_path, load_config
from delivery_rl.envs.pedestrians import PedestrianManager
from delivery_rl.envs.robot import MecanumRobot
from delivery_rl.envs.safety_shield import SafetyShield, CLEAR
from delivery_rl.envs.sensors import SensorSuite
from delivery_rl.envs.world import CorridorWorld
from delivery_rl.tasks.delivery_task import DeliveryTask

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # delivery_rl/
_A_MAX = 5.0  # m/s^2 used to normalise IMU acceleration into [-1, 1]


class CorridorDeliveryEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 20}

    def __init__(self, config: Optional[dict] = None, config_path: Optional[str] = None,
                 render_mode: Optional[str] = None):
        super().__init__()
        if config is None:
            config = load_config(config_path or default_config_path())
        self.config = config
        self.render_mode = render_mode
        self.base_dir = _BASE_DIR

        gui = (render_mode == "human") or bool(config["env"].get("gui", False))
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(float(config["env"]["sim_dt"]), physicsClientId=self.client)

        self.rng = np.random.default_rng(int(config.get("seed", 0)))

        # Build the (static) world once so we know the locker count for the space.
        self.world = CorridorWorld(self.client, config, self.rng, self.base_dir)
        self.scene = self.world.build()
        self.robot = MecanumRobot(self.client, config, self.rng, self.base_dir)
        self.robot.load()
        self.sensors = SensorSuite(self.client, config, self.rng)
        self.pedestrians = PedestrianManager(self.client, config, self.rng)
        self.task = DeliveryTask(config, self.rng)

        self.num_lockers = len(self.scene.lockers)
        self.half_len = self.scene.corridor_length / 2.0
        _wcfg = config["env"]["world"]
        # normalisation span for odometry y (covers rooms / arc, not just corridor)
        ymin, ymax = self.scene.bounds[2], self.scene.bounds[3]
        self.odom_y_norm = max(abs(ymin), abs(ymax), self.scene.corridor_width)
        self.dock_align_yaw = float(config["env"]["mechanism"]["dock_align_yaw"])
        self.max_steps = int(config["env"]["max_episode_steps"])
        self.tilt_threshold = float(config["env"]["termination"]["tilt_threshold_deg"])

        # global waypoint planner (guides the reactive policy around rooms / arc)
        self.use_planner = bool(_wcfg.get("use_planner", True))
        self.waypoint_reach = float(_wcfg.get("waypoint_reach_dist", 0.45))
        robot_r = max(self.robot.base_size[0], self.robot.base_size[1]) / 2.0
        if self.use_planner:
            from delivery_rl.envs.planner import GridPlanner
            self.planner = GridPlanner(self.world, robot_r)
        else:
            self.planner = None
        self.waypoints: list = []
        self.wp_idx = 0
        self._planned_for = None   # locker id the current waypoints lead to

        # Execution-layer helpers (do not change training): action smoothing to
        # reduce vibration, and a reactive pedestrian-avoidance safety shield.
        self.action_smoothing = float(config["env"].get("control", {}).get("action_smoothing", 0.0))
        robot_radius = max(self.robot.base_size[0], self.robot.base_size[1]) / 2.0
        self.shield = SafetyShield(config, robot_radius, self.scene.corridor_width / 2.0)
        self.ped_nonterminal = bool(
            config["env"].get("safety", {}).get("pedestrian_nonterminal_when_yielding", True))
        self.smooth_action = np.zeros(3, dtype=np.float32)
        self.last_shield_status = "clear"
        self.last_beep = False
        self.target_marker_id = -1   # red dot above the current target locker
        self.beep_marker_id = -1     # red ball above the robot while beeping

        obs_dim = (self.sensors.total_dim + 3 + self.num_lockers + 1 + 5)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

        self.episode_step = 0
        self.sim_time = 0.0
        self.prev_target_dist = 0.0
        self.pending_locker: Optional[int] = None

    # ------------------------------------------------------------------ #
    def _replan(self, robot_xy) -> None:
        """(Re)compute waypoints from the robot to the current locker dock."""
        ttype, goal = self.task.current_target_xy(np.asarray(robot_xy, dtype=np.float32))
        cur_target = self.task.current_target_locker(np.asarray(robot_xy, dtype=np.float32))
        cur_id = cur_target.id if cur_target is not None else "dock"
        if not self.use_planner:
            self.waypoints = [tuple(goal)]
            self.wp_idx = 0
            self._planned_for = cur_id
            return
        path = self.planner.plan(tuple(robot_xy), tuple(goal))
        wps = path if path else [tuple(goal)]
        # For room / arc lockers, insert the mouth-centre as a waypoint just before
        # the dock so the robot enters straight through the opening instead of
        # cutting the corner and clipping a side wall (a terminal collision).
        if cur_target is not None and cur_target.side in ("room", "arc") and len(wps) >= 1:
            mouth = self._mouth_point(cur_target)
            if mouth is not None:
                wps = wps[:-1] + [mouth, wps[-1]]
        self.waypoints = wps
        self.wp_idx = 0
        self._planned_for = cur_id

    def _mouth_point(self, locker):
        """A point centred in the corridor lane just outside the room/arc opening,
        so the approach is perpendicular to the doorway."""
        hw = self.scene.corridor_width / 2.0
        if locker.side == "room":
            # room opens off the north/south wall at x = locker x; aim at the lane
            sign = 1.0 if locker.pos[1] > 0 else -1.0
            return (float(locker.pos[0]), float(sign * (hw - 0.4)))
        if locker.side == "arc":
            # aim at the corridor lane above the arc centre before descending
            return (float(locker.dock[0]), float(-(hw - 0.4)))
        return None

    def _nav_target(self, robot_xy) -> np.ndarray:
        """Next waypoint to steer toward (advances as the robot reaches each).

        Falls back to the locker dock when planning is off or finished. This is
        what the policy observes as its relative-pose target, so it follows the
        corridor/arc around bends instead of driving straight into a wall."""
        rxy = np.asarray(robot_xy, dtype=np.float32)
        cur_target = self.task.current_target_locker(rxy)
        cur_id = cur_target.id if cur_target is not None else "dock"
        if cur_id != self._planned_for or not self.waypoints:
            self._replan(rxy)
        # advance through reached waypoints (keep the last as the goal)
        while self.wp_idx < len(self.waypoints) - 1:
            wp = np.asarray(self.waypoints[self.wp_idx], dtype=np.float32)
            if float(np.linalg.norm(wp - rxy)) <= self.waypoint_reach:
                self.wp_idx += 1
            else:
                break
        return np.asarray(self.waypoints[self.wp_idx], dtype=np.float32)

    # ------------------------------------------------------------------ #
    def _reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        for obj in (self.world, self.robot, self.sensors, self.pedestrians, self.task):
            obj.np_random = self.rng

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._reseed(seed)

        self.task.reset(self.scene)
        settings = self.task.episode_settings()
        self.world.reset_obstacles(settings["num_obstacles"])
        self.sensors.noise_scale = settings["noise_scale"]

        dock = self.scene.dock_pos
        sx, sy, syaw = dock[0], dock[1], 0.0
        if settings["domain_random"]:
            sx += float(self.rng.uniform(-0.3, 0.3))
            sy += float(self.rng.uniform(-0.3, 0.3))
            syaw = float(self.rng.uniform(-math.pi, math.pi))
            p.changeDynamics(self.robot.body_id, -1,
                             mass=self.robot.base_mass * float(self.rng.uniform(0.8, 1.2)),
                             physicsClientId=self.client)
        self.robot.reset(sx, sy, syaw)
        # spawn residents between the robot and its target so their paths cross
        _, tgt0 = self.task.current_target_xy(np.array([sx, sy], dtype=np.float32))
        self.pedestrians.reset(settings["num_pedestrians"], robot_xy=(sx, sy),
                               target_xy=(float(tgt0[0]), float(tgt0[1])))

        self.episode_step = 0
        self.sim_time = 0.0
        self.pending_locker = None
        self.smooth_action[:] = 0.0
        self.last_shield_status = "clear"
        self.last_beep = False
        self.shield.reset()
        rxy = np.array(self.robot.get_pose()[:2], dtype=np.float32)
        self._planned_for = None
        self.waypoints = []
        self._nav_target(rxy)   # build the initial waypoint plan
        # progress reward tracks distance to the LOCKER goal (consistent with step)
        _, goal = self.task.current_target_xy(rxy)
        self.prev_target_dist = float(np.linalg.norm(goal - rxy))

        obs = self._build_obs()
        info = {"manifest": [(m.parcel, m.locker_id) for m in self.task.manifest],
                "num_parcels": len(self.task.manifest)}
        return obs, info

    # ------------------------------------------------------------------ #
    def step(self, action):
        self.episode_step += 1
        self.sim_time += self.robot.dt
        a_raw = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        # low-pass smoothing reduces jerky steering / vibration (execution-time
        # only; does not affect how the policy was trained)
        alpha = self.action_smoothing
        self.smooth_action = alpha * self.smooth_action + (1.0 - alpha) * a_raw
        a = self.smooth_action.copy()

        rxy = np.array(self.robot.get_pose()[:2], dtype=np.float32)
        prev_type, prev_locker_goal = self.task.current_target_xy(rxy)
        prev_tgt = self._nav_target(rxy)   # next waypoint (for the shield's "ahead" dir)

        # reactive pedestrian-avoidance shield: override the command near people
        self.last_shield_status = "clear"
        self.last_beep = False
        x0, y0, yaw0 = self.robot.get_pose()
        if self.pedestrians.body_ids and not self.robot.macro_active:
            ped_states = self.pedestrians.get_states((x0, y0))
            a, status, beep = self.shield.filter(a, (x0, y0), yaw0, prev_tgt, ped_states)
            self.last_shield_status = status
            self.last_beep = beep

        vx = float(a[0]) * self.robot.max_linear_speed
        vy = float(a[1]) * self.robot.max_linear_speed
        omega = float(a[2]) * self.robot.max_yaw_rate

        delivered = wrong = all_done = returned = False
        move = {"distance_moved": 0.0, "d_omega": 0.0, "collision_world": False,
                "collision_pedestrian": False, "tilt_deg": 0.0}

        if self.robot.macro_active:
            if self.robot.step_macro():  # macro finished this step
                lid = self.pending_locker
                self.pending_locker = None
                if lid is not None and self.task.deliver(lid):
                    delivered = True
                    self.robot.advance_carousel()
                    all_done = self.task.all_delivered
                else:
                    wrong = True
        else:
            move = self.robot.apply_velocity(
                vx, vy, omega, self.world.all_collision_ids,
                self.pedestrians.body_ids, self.world.floor_id)
            # With the safety shield enabled, contact with a resident is never
            # mission-ending: apply_velocity already reverts on contact (no
            # penetration), so the robot simply stops at the person and beeps,
            # exactly the requested "if a person blocks it, stop and beep"
            # behaviour. (Static-world collisions remain terminal.)
            if (self.ped_nonterminal and self.shield.enabled
                    and move.get("collision_pedestrian")):
                move["collision_pedestrian"] = False
                self.last_beep = True
            if self.task.parcels_carried > 0 and not self.robot.macro_active:
                rxy2 = np.array(self.robot.get_pose()[:2], dtype=np.float32)
                lid = self.task.docked_locker_id(rxy2)
                # Navigation phase: dock by POSITION (robot inside the target
                # locker's dock zone). Yaw-alignment ("canh khay") is an optional
                # refinement kept as a future macro-learning hook; enable it by
                # setting mechanism.dock_align_yaw < pi and flipping the flag below.
                require_alignment = self.dock_align_yaw < math.pi
                if lid is not None and (not require_alignment or self._aligned_to_locker(lid)):
                    self.robot.start_macro()
                    self.pending_locker = lid

        if self.pedestrians.body_ids:
            self.pedestrians.step(self.sim_time)

        rxy2 = np.array(self.robot.get_pose()[:2], dtype=np.float32)
        cur_type, cur_tgt = self.task.current_target_xy(rxy2)
        cur_dist = float(np.linalg.norm(cur_tgt - rxy2))
        if delivered or wrong or cur_type != prev_type:
            progress = 0.0           # target switched -> no spurious progress jump
        else:
            progress = self.prev_target_dist - cur_dist
        self.prev_target_dist = cur_dist

        if self.task.all_delivered and self.task.at_dock(rxy2):
            returned = True
        tilt = move.get("tilt_deg", 0.0) > self.tilt_threshold

        events = {
            "progress": progress,
            "distance_moved": move.get("distance_moved", 0.0),
            "d_omega": move.get("d_omega", 0.0),
            "collision_world": move.get("collision_world", False),
            "collision_pedestrian": move.get("collision_pedestrian", False),
            "delivered": delivered, "wrong_delivery": wrong,
            "all_done": all_done, "returned_dock": returned, "tilt": tilt,
        }
        reward = self.task.compute_reward(events)
        terminated, outcome = self.task.check_termination(events)
        truncated = self.episode_step >= self.max_steps

        obs = self._build_obs()
        info = {
            "outcome": outcome,
            "delivered": delivered,
            "wrong_delivery": wrong,
            "collision": events["collision_world"] or events["collision_pedestrian"],
            "deliveries_done": sum(m.delivered for m in self.task.manifest),
            "num_parcels": len(self.task.manifest),
            "parcels_remaining": self.task.parcels_carried,
            "returned_dock": returned,
            "is_success": outcome == "success",
            "shield_status": self.last_shield_status,
            "beep": self.last_beep,
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    def _aligned_to_locker(self, lid: int) -> bool:
        if self.dock_align_yaw >= math.pi:
            return True
        locker = self.task.locker_by_id(lid)
        x, y, yaw = self.robot.get_pose()
        bearing = math.atan2(locker.pos[1] - y, locker.pos[0] - x)
        return abs(self._wrap(bearing - yaw)) <= self.dock_align_yaw

    def _build_obs(self) -> np.ndarray:
        x, y, yaw = self.robot.get_pose()
        vx, vy, omega = self.robot.body_velocity()
        pos = (x, y)

        lidar = self.sensors.read_lidar(pos, yaw)
        tof = self.sensors.read_tof(pos, yaw)
        imu = self.sensors.read_imu(self.robot.accel_body, omega, yaw)
        odom = self.sensors.read_odometry(x, y, yaw, vx, vy, omega)

        imu_n = np.array([
            imu[0] / _A_MAX, imu[1] / _A_MAX, imu[2] / _A_MAX,
            imu[3] / self.robot.max_yaw_rate, imu[4] / self.robot.max_yaw_rate,
            imu[5] / self.robot.max_yaw_rate, imu[6] / math.pi], dtype=np.float32)
        odom_n = np.array([
            odom[0] / self.half_len, odom[1] / self.odom_y_norm, odom[2] / math.pi,
            odom[3] / self.robot.max_linear_speed, odom[4] / self.robot.max_linear_speed,
            odom[5] / self.robot.max_yaw_rate], dtype=np.float32)

        rxy = np.array(pos, dtype=np.float32)
        # the policy steers toward the next WAYPOINT (planner) -> follows the
        # corridor / arc around bends; without a planner this is the locker dock.
        tgt = self._nav_target(rxy)
        dvec = tgt - rxy
        c, s = math.cos(yaw), math.sin(yaw)
        dxb = c * dvec[0] + s * dvec[1]
        dyb = -s * dvec[0] + c * dvec[1]
        bearing = self._wrap(math.atan2(dvec[1], dvec[0]) - yaw)
        rel = np.array([dxb / self.half_len, dyb / self.half_len, bearing / math.pi],
                       dtype=np.float32)

        remaining = self.task.remaining_vector()
        carried = np.array([self.task.parcels_carried / max(len(self.task.manifest), 1)],
                           dtype=np.float32)
        mech = self.robot.mechanism_obs()

        obs = np.concatenate([lidar, tof, imu_n, odom_n, rel, remaining, carried, mech])
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------ #
    def _ensure_markers(self) -> None:
        if self.target_marker_id < 0:
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.13,
                                      rgbaColor=[1.0, 0.05, 0.05, 1.0],
                                      physicsClientId=self.client)
            self.target_marker_id = p.createMultiBody(
                baseMass=0.0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis,
                basePosition=[0, 0, -5], physicsClientId=self.client)
        if self.beep_marker_id < 0:
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.16,
                                      rgbaColor=[1.0, 0.2, 0.0, 0.9],
                                      physicsClientId=self.client)
            self.beep_marker_id = p.createMultiBody(
                baseMass=0.0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis,
                basePosition=[0, 0, -5], physicsClientId=self.client)

    def _update_markers(self) -> None:
        """Red dot hovers over the locker the robot is currently routing to."""
        self._ensure_markers()
        rxy = np.array(self.robot.get_pose()[:2], dtype=np.float32)
        locker = self.task.current_target_locker(rxy)
        if locker is not None:
            top = locker.pos[2] + self.config["env"]["world"]["locker_size"][2] / 2 + 0.25
            p.resetBasePositionAndOrientation(
                self.target_marker_id, [locker.pos[0], locker.pos[1], top],
                [0, 0, 0, 1], physicsClientId=self.client)
        else:
            p.resetBasePositionAndOrientation(self.target_marker_id, [0, 0, -5],
                                              [0, 0, 0, 1], physicsClientId=self.client)
        x, y, _ = self.robot.get_pose()
        bz = 0.7 if self.last_beep else -5.0
        p.resetBasePositionAndOrientation(self.beep_marker_id, [x, y, bz],
                                          [0, 0, 0, 1], physicsClientId=self.client)

    def render(self):
        """Synthetic overhead camera -- for human viewing ONLY, never observed."""
        if self.render_mode not in ("rgb_array", "human"):
            return None
        self._update_markers()
        x, y, yaw = self.robot.get_pose()
        view = p.computeViewMatrix(
            cameraEyePosition=[x - 2.5 * math.cos(yaw), y - 2.5 * math.sin(yaw), 3.0],
            cameraTargetPosition=[x, y, 0.0], cameraUpVector=[0, 0, 1],
            physicsClientId=self.client)
        proj = p.computeProjectionMatrixFOV(60, 4 / 3, 0.1, 40, physicsClientId=self.client)
        w, h = 320, 240
        img = p.getCameraImage(w, h, view, proj, renderer=p.ER_TINY_RENDERER,
                               physicsClientId=self.client)
        rgb = np.reshape(np.asarray(img[2], dtype=np.uint8), (h, w, 4))[:, :, :3]
        return rgb

    def close(self):
        if self.client is not None and p.isConnected(self.client):
            p.disconnect(self.client)
            self.client = None

    @staticmethod
    def _wrap(angle: float) -> float:
        return (angle + math.pi) % (2 * math.pi) - math.pi
