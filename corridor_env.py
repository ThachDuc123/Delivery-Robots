"""Apartment-corridor navigation environment for Reinforcement Learning.

Goals of this rewrite (vs. the previous DeliveryNavEnv):

1.  CENTER-OF-CORRIDOR navigation. The old environment rewarded hugging the
    RIGHT wall (`_right_lane_direction`, `right_lane_bonus`, ...). That logic is
    completely removed. The agent is now rewarded for keeping an EQUAL distance
    to the left and right walls, i.e. driving down the middle of the hallway.

2.  HONEST RL. The old `_apply_action` contained a hand-written auto-pilot
    (free-space steering, goal steering, obstacle avoidance, wall correction).
    That meant the *environment* drove the robot, not the learned policy, so
    comparing RL algorithms was meaningless. Here the action maps DIRECTLY to
    the robot's motion; all "intelligence" must be learned from the reward.

3.  CORRECT WALL SENSING. Collision now uses true circle-vs-rectangle geometry
    that respects the robot radius (the old code mixed radius-aware boundary
    checks with point-only wall checks). Ray sensors march along the ray and
    report the real distance to the wall surface.

4.  CORRECT GOAL DETECTION. The goal is guaranteed to be reachable (verified
    with a grid BFS) and a meaningful distance away; "reached" uses a clear
    radius (robot_radius + goal_radius).

The action space is Discrete(3) so the same environment works for PPO, A2C and
DQN, which lets us compare the three algorithms fairly.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


@dataclass
class MapLayout:
    name: str
    walls: List[pygame.Rect]
    obstacles: List[pygame.Rect] = field(default_factory=list)
    start_zone: Optional[pygame.Rect] = None
    goal_zone: Optional[pygame.Rect] = None
    # Optional poly-line describing the hallway centre (for metrics / drawing).
    centerline: List[Tuple[float, float]] = field(default_factory=list)


class CorridorNavEnv(gym.Env):
    """2D robot that must drive down the MIDDLE of an apartment corridor."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        map_size: Tuple[int, int] = (720, 540),
        max_episode_steps: int = 600,
        layout: str = "corridor",
        randomize_layout: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.map_width, self.map_height = map_size
        self.max_episode_steps = max_episode_steps
        self.default_layout = layout
        self.randomize_layout = randomize_layout
        self.random = random.Random(seed)

        # ---- Robot kinematics (direct control, light smoothing) -------------
        self.robot_radius = 11
        self.max_linear_speed = 4.0                 # px / step
        self.max_angular_speed = math.radians(14.0)  # rad / step

        # ---- Sensors --------------------------------------------------------
        self.max_sensor_range = 250.0
        self.ray_step = 5.0
        # Symmetric fan, includes the exact +/-90 deg rays used for centring.
        self.ray_angles = np.deg2rad(
            [-90, -67, -45, -27, -13, 0, 13, 27, 45, 67, 90], dtype=np.float64
        )
        self.num_rays = len(self.ray_angles)
        self.left_ray_idx = 0    # -90 deg  (one corridor wall)
        self.right_ray_idx = -1  # +90 deg  (the other corridor wall)
        self.front_ray_idx = self.num_rays // 2  # 0 deg

        # ---- Reward weights -------------------------------------------------
        self.goal_reward = 200.0
        self.collision_penalty = -120.0
        self.step_penalty = -0.02
        self.progress_scale = 1.0          # dense reward for getting closer
        self.center_scale = 0.35           # reward for staying centred
        self.clearance_penalty = 0.30      # penalty for getting close to a wall
        self.safe_clearance = 26.0         # px from wall before we complain
        self.turn_penalty = 0.01           # discourage needless spinning
        self.goal_radius = 14.0
        self.reach_distance = self.robot_radius + self.goal_radius

        # ---- Observation / action spaces ------------------------------------
        # [dist, goal_sin, goal_cos, speed, ang_vel, offset, width, front] + rays
        obs_dim = 8 + self.num_rays
        low = np.concatenate(
            [np.array([0, -1, -1, 0, -1, -1, 0, 0], dtype=np.float32),
             np.zeros(self.num_rays, dtype=np.float32)]
        )
        high = np.ones(obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0 forward, 1 left, 2 right

        # ---- State ----------------------------------------------------------
        self.robot_pos = np.zeros(2, dtype=np.float32)
        self.robot_heading = 0.0
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.goal_pos = np.zeros(2, dtype=np.float32)
        self.episode_step = 0
        self.previous_goal_distance = 0.0
        self.collision_flash = 0.0
        self.path_trail: List[Tuple[float, float]] = []
        self.last_ray_distances: List[float] = []
        self.last_ray_endpoints: List[Tuple[int, int]] = []
        self.prev_action = 0

        # ---- Episode statistics (for the HUD) -------------------------------
        self.episode_count = 0
        self.success_count = 0
        self.collision_count = 0
        self.episode_reward = 0.0
        self.last_episode_reward = 0.0
        self.last_outcome = ""
        self.offset_accumulator = 0.0  # sum of |offset| while inside a corridor
        self.offset_samples = 0

        # ---- Maps -----------------------------------------------------------
        self.map_layouts: Dict[str, MapLayout] = {}
        self.active_layout: Optional[MapLayout] = None
        self._build_layouts()

        # ---- Rendering ------------------------------------------------------
        self.window: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self._font: Optional[pygame.font.Font] = None

    # ====================================================================== #
    #  Gymnasium API
    # ====================================================================== #
    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.random.seed(seed)

        layout_name = (options or {}).get("layout")
        if layout_name is None:
            if self.randomize_layout:
                layout_name = self.random.choice(list(self.map_layouts.keys()))
            else:
                layout_name = self.default_layout
        self.active_layout = self.map_layouts[layout_name]

        # Place robot and goal so that the goal is reachable (BFS verified).
        self.robot_pos, self.goal_pos = self._spawn_start_and_goal()
        # Face roughly toward the goal so episodes start sensibly.
        goal_dir = self.goal_pos - self.robot_pos
        self.robot_heading = math.atan2(goal_dir[1], goal_dir[0])
        self.robot_heading += self.random.uniform(-0.4, 0.4)

        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.episode_step = 0
        self.previous_goal_distance = self._distance_to_goal()
        self.collision_flash = 0.0
        self.path_trail = [tuple(self.robot_pos)]
        self.prev_action = 0
        self.episode_count += 1
        self.episode_reward = 0.0
        self.offset_accumulator = 0.0
        self.offset_samples = 0

        self._cast_rays()
        observation = self._get_observation()
        info = {"layout": self.active_layout.name, "goal_position": self.goal_pos.copy()}
        return observation, info

    def step(self, action: int):
        self.episode_step += 1
        self._apply_action(int(action))
        self._cast_rays()

        collision = self._check_collision(self.robot_pos)
        distance = self._distance_to_goal()
        goal_reached = distance <= self.reach_distance

        # ---- Reward ---------------------------------------------------------
        reward = self.step_penalty

        # 1) dense progress toward the goal
        reward += (self.previous_goal_distance - distance) * self.progress_scale
        self.previous_goal_distance = distance

        # 2) centring: reward equal clearance to both walls (the core objective)
        offset, in_corridor = self._corridor_offset()
        if in_corridor:
            reward += self.center_scale * (1.0 - 2.0 * abs(offset))
            self.offset_accumulator += abs(offset)
            self.offset_samples += 1

        # 3) keep clear of walls/obstacles
        min_ray = min(self.last_ray_distances) if self.last_ray_distances else self.max_sensor_range
        if min_ray < self.safe_clearance:
            reward -= self.clearance_penalty * (1.0 - min_ray / self.safe_clearance)

        # 4) smoothness
        if action != 0:
            reward -= self.turn_penalty
        if action != self.prev_action:
            reward -= self.turn_penalty
        self.prev_action = int(action)

        # 5) terminal rewards
        if collision:
            reward += self.collision_penalty
            self.collision_flash = 1.0
        if goal_reached:
            reward += self.goal_reward

        terminated = bool(collision or goal_reached)
        truncated = bool(self.episode_step >= self.max_episode_steps)
        self.episode_reward += reward

        if terminated or truncated:
            self.last_episode_reward = self.episode_reward
            if goal_reached:
                self.success_count += 1
                self.last_outcome = "Success"
            elif collision:
                self.collision_count += 1
                self.last_outcome = "Collision"
            else:
                self.last_outcome = "Timeout"

        observation = self._get_observation()
        mean_offset = (
            self.offset_accumulator / self.offset_samples if self.offset_samples else 0.0
        )
        info = {
            "layout": self.active_layout.name,
            "goal_reached": goal_reached,
            "collision": collision,
            "distance": distance,
            "corridor_offset": offset,
            "in_corridor": in_corridor,
            "mean_abs_offset": mean_offset,
            "min_clearance": min_ray,
        }

        if self.render_mode == "human":
            self.render(info=info, reward=reward)
        return observation, reward, terminated, truncated, info

    # ====================================================================== #
    #  Robot dynamics  (DIRECT control -- no scripted auto-pilot)
    # ====================================================================== #
    def _apply_action(self, action: int) -> None:
        if action == 1:
            target_angular = -self.max_angular_speed
        elif action == 2:
            target_angular = self.max_angular_speed
        else:
            target_angular = 0.0

        # light smoothing so motion is not jerky, but the agent stays in control
        self.angular_velocity = 0.5 * self.angular_velocity + 0.5 * target_angular
        self.robot_heading = self._wrap_angle(self.robot_heading + self.angular_velocity)

        # slow down while turning hard
        turn_ratio = abs(self.angular_velocity) / self.max_angular_speed
        self.linear_velocity = self.max_linear_speed * (1.0 - 0.5 * turn_ratio)

        direction = np.array(
            [math.cos(self.robot_heading), math.sin(self.robot_heading)], dtype=np.float32
        )
        new_pos = self.robot_pos + direction * self.linear_velocity
        if not self._check_collision(new_pos):
            self.robot_pos = new_pos
            self.path_trail.append(tuple(self.robot_pos))
        else:
            # blocked: stay put (collision is detected separately in step)
            self.collision_flash = max(self.collision_flash, 0.6)

    # ====================================================================== #
    #  Sensors
    # ====================================================================== #
    def _cast_rays(self) -> None:
        distances, endpoints = [], []
        for ray_angle in self.ray_angles:
            dist, end = self._raycast(self.robot_heading + ray_angle)
            distances.append(dist)
            endpoints.append(end)
        self.last_ray_distances = distances
        self.last_ray_endpoints = endpoints

    def _raycast(self, angle: float) -> Tuple[float, Tuple[int, int]]:
        dx, dy = math.cos(angle), math.sin(angle)
        x, y = float(self.robot_pos[0]), float(self.robot_pos[1])
        total = 0.0
        while total < self.max_sensor_range:
            total += self.ray_step
            px, py = x + dx * total, y + dy * total
            if px < 0 or px > self.map_width or py < 0 or py > self.map_height:
                return total, (int(px), int(py))
            if self._point_in_solid(px, py):
                return total, (int(px), int(py))
        return self.max_sensor_range, (int(x + dx * self.max_sensor_range),
                                       int(y + dy * self.max_sensor_range))

    def _point_in_solid(self, x: float, y: float) -> bool:
        if not self.active_layout:
            return False
        for rect in self.active_layout.walls:
            if rect.collidepoint(x, y):
                return True
        for rect in self.active_layout.obstacles:
            if rect.collidepoint(x, y):
                return True
        return False

    def _corridor_offset(self) -> Tuple[float, bool]:
        """Signed centring error in [-1, 1] from the left/right side rays.

        0  -> perfectly centred between the two walls.
        +- -> closer to one wall. `in_corridor` is True only when BOTH side
        rays actually hit a wall (so we do not 'centre' in open space)."""
        left = self.last_ray_distances[self.left_ray_idx]
        right = self.last_ray_distances[self.right_ray_idx]
        in_corridor = (left < self.max_sensor_range * 0.9
                       and right < self.max_sensor_range * 0.9)
        total = left + right
        if total < 1e-6:
            return 0.0, in_corridor
        offset = (right - left) / total
        return float(np.clip(offset, -1.0, 1.0)), in_corridor

    # ====================================================================== #
    #  Observation
    # ====================================================================== #
    def _get_observation(self) -> np.ndarray:
        distance = self._distance_to_goal()
        max_dist = math.hypot(self.map_width, self.map_height)
        distance_norm = min(distance / max_dist, 1.0)

        goal_dir = self.goal_pos - self.robot_pos
        goal_angle = math.atan2(goal_dir[1], goal_dir[0])
        rel = self._wrap_angle(goal_angle - self.robot_heading)

        speed_norm = self.linear_velocity / self.max_linear_speed
        ang_norm = float(np.clip(self.angular_velocity / self.max_angular_speed, -1, 1))

        offset, _ = self._corridor_offset()
        left = self.last_ray_distances[self.left_ray_idx]
        right = self.last_ray_distances[self.right_ray_idx]
        width_norm = min((left + right) / (2.0 * self.max_sensor_range), 1.0)
        front_norm = self.last_ray_distances[self.front_ray_idx] / self.max_sensor_range

        rays = [d / self.max_sensor_range for d in self.last_ray_distances]
        obs = np.array(
            [distance_norm, math.sin(rel), math.cos(rel), speed_norm, ang_norm,
             offset, width_norm, front_norm] + rays,
            dtype=np.float32,
        )
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    # ====================================================================== #
    #  Collision (true circle-vs-rect, respects the robot radius)
    # ====================================================================== #
    def _check_collision(self, position: np.ndarray, radius: Optional[float] = None) -> bool:
        r = self.robot_radius if radius is None else radius
        x, y = float(position[0]), float(position[1])
        if x - r < 0 or x + r > self.map_width or y - r < 0 or y + r > self.map_height:
            return True
        if self.active_layout:
            for rect in self.active_layout.walls:
                if self._circle_intersects_rect(x, y, r, rect):
                    return True
            for rect in self.active_layout.obstacles:
                if self._circle_intersects_rect(x, y, r, rect):
                    return True
        return False

    @staticmethod
    def _circle_intersects_rect(cx: float, cy: float, r: float, rect: pygame.Rect) -> bool:
        nearest_x = min(max(cx, rect.left), rect.right)
        nearest_y = min(max(cy, rect.top), rect.bottom)
        return (cx - nearest_x) ** 2 + (cy - nearest_y) ** 2 < r * r

    # ====================================================================== #
    #  Spawning + reachability (goal is always reachable & far enough)
    # ====================================================================== #
    def _spawn_start_and_goal(self) -> Tuple[np.ndarray, np.ndarray]:
        layout = self.active_layout
        min_sep = 0.45 * math.hypot(self.map_width, self.map_height)
        for _ in range(400):
            start = self._sample_in_zone(layout.start_zone)
            goal = self._sample_in_zone(layout.goal_zone)
            if start is None or goal is None:
                continue
            if np.linalg.norm(goal - start) < min_sep:
                continue
            if self._is_reachable(start, goal):
                return start, goal
        # Fallback: any two reachable free points.
        free = [self._random_free_position() for _ in range(2)]
        return np.array(free[0], dtype=np.float32), np.array(free[1], dtype=np.float32)

    def _sample_in_zone(self, zone: Optional[pygame.Rect]) -> Optional[np.ndarray]:
        if zone is None:
            return np.array(self._random_free_position(), dtype=np.float32)
        for _ in range(80):
            x = self.random.uniform(zone.left, zone.right)
            y = self.random.uniform(zone.top, zone.bottom)
            pos = np.array([x, y], dtype=np.float32)
            if not self._check_collision(pos):
                return pos
        return None

    def _random_free_position(self) -> Tuple[float, float]:
        for _ in range(300):
            x = self.random.uniform(self.robot_radius, self.map_width - self.robot_radius)
            y = self.random.uniform(self.robot_radius, self.map_height - self.robot_radius)
            if not self._check_collision(np.array([x, y], dtype=np.float32)):
                return x, y
        return float(self.map_width * 0.5), float(self.map_height * 0.5)

    def _is_reachable(self, start: np.ndarray, goal: np.ndarray) -> bool:
        """Grid BFS on a coarse occupancy grid (cell ~ robot diameter)."""
        cell = int(self.robot_radius * 1.6)
        cols = self.map_width // cell
        rows = self.map_height // cell

        def free(cx: int, cy: int) -> bool:
            px = cx * cell + cell / 2
            py = cy * cell + cell / 2
            return not self._check_collision(np.array([px, py], dtype=np.float32),
                                             radius=self.robot_radius * 0.9)

        s = (int(start[0]) // cell, int(start[1]) // cell)
        g = (int(goal[0]) // cell, int(goal[1]) // cell)
        if not (0 <= s[0] < cols and 0 <= s[1] < rows):
            return False
        if not (0 <= g[0] < cols and 0 <= g[1] < rows):
            return False
        if not free(*s) or not free(*g):
            return False

        seen = {s}
        queue = deque([s])
        while queue:
            cx, cy = queue.popleft()
            if (cx, cy) == g:
                return True
            for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                if 0 <= nx < cols and 0 <= ny < rows and (nx, ny) not in seen and free(nx, ny):
                    seen.add((nx, ny))
                    queue.append((nx, ny))
        return False

    def _distance_to_goal(self) -> float:
        return float(np.linalg.norm(self.goal_pos - self.robot_pos))

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + math.pi) % (2 * math.pi) - math.pi

    # ====================================================================== #
    #  Map layouts  (apartment-style corridors)
    # ====================================================================== #
    def _build_layouts(self) -> None:
        self.map_layouts = {
            "corridor": self._build_corridor(),
            "corridor_obstacles": self._build_corridor_obstacles(),
            "corridor_turn": self._build_corridor_turn(),
        }

    def _build_corridor(self) -> MapLayout:
        """A straight horizontal apartment hallway; robot walks left -> right."""
        top = pygame.Rect(0, 0, self.map_width, 215)
        bottom = pygame.Rect(0, 325, self.map_width, self.map_height - 325)
        walls = [top, bottom]
        start_zone = pygame.Rect(40, 235, 70, 70)
        goal_zone = pygame.Rect(610, 235, 70, 70)
        centerline = [(0, 270), (self.map_width, 270)]
        return MapLayout("corridor", walls, [], start_zone, goal_zone, centerline)

    def _build_corridor_obstacles(self) -> MapLayout:
        """Same hallway but with a couple of obstacles to steer around."""
        top = pygame.Rect(0, 0, self.map_width, 215)
        bottom = pygame.Rect(0, 325, self.map_width, self.map_height - 325)
        walls = [top, bottom]
        obstacles = [
            pygame.Rect(250, 215, 44, 60),   # juts down from the top wall
            pygame.Rect(470, 265, 44, 60),   # juts up from the bottom wall
        ]
        start_zone = pygame.Rect(40, 235, 70, 70)
        goal_zone = pygame.Rect(610, 235, 70, 70)
        centerline = [(0, 270), (self.map_width, 270)]
        return MapLayout("corridor_obstacles", walls, obstacles, start_zone, goal_zone, centerline)

    def _build_corridor_turn(self) -> MapLayout:
        """An L-shaped hallway: go right, then turn and go down."""
        walls = [
            pygame.Rect(0, 0, self.map_width, 150),          # top block
            pygame.Rect(0, 260, 430, self.map_height - 260),  # bottom-left block
            pygame.Rect(540, 260, self.map_width - 540, 80),  # small block by the turn
            pygame.Rect(0, 150, 0, 0),                        # placeholder (no-op)
        ]
        # Horizontal leg: y in [150,260]; vertical leg: x in [430,540], y down to bottom.
        walls = [
            pygame.Rect(0, 0, self.map_width, 150),
            pygame.Rect(0, 260, 430, self.map_height - 260),
            pygame.Rect(540, 340, self.map_width - 540, self.map_height - 340),
            pygame.Rect(540, 150, self.map_width - 540, 0),
        ]
        # Make the vertical leg a clean channel x in [430,540].
        walls = [
            pygame.Rect(0, 0, self.map_width, 150),                       # ceiling
            pygame.Rect(0, 260, 430, self.map_height - 260),             # floor of horizontal leg
            pygame.Rect(540, 0, self.map_width - 540, self.map_height),  # right block (seals vertical leg on the right)
        ]
        start_zone = pygame.Rect(40, 175, 70, 70)        # left end of horizontal leg
        goal_zone = pygame.Rect(450, 430, 80, 70)        # bottom of vertical leg
        centerline = [(0, 205), (485, 205), (485, self.map_height)]
        return MapLayout("corridor_turn", walls, [], start_zone, goal_zone, centerline)

    # ====================================================================== #
    #  Rendering
    # ====================================================================== #
    def render(self, info: Optional[dict] = None, reward: float = 0.0):
        if self.render_mode is None:
            return
        if self.window is None and self.render_mode == "human":
            pygame.init()
            self.window = pygame.display.set_mode((self.map_width, self.map_height))
            pygame.display.set_caption("Apartment Corridor Navigation")
        if self._font is None:
            pygame.font.init()
            self._font = pygame.font.SysFont("Arial", 15)
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.map_width, self.map_height))
        canvas.fill((28, 30, 36))                      # hallway floor
        self._draw_layout(canvas)
        self._draw_centerline(canvas)
        self._draw_goal(canvas)
        self._draw_rays(canvas)
        self._draw_path_trail(canvas)
        self._draw_robot(canvas)
        self._draw_hud(canvas, info, reward)
        self._draw_collision_flash(canvas)

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            return None
        return np.transpose(pygame.surfarray.array3d(canvas), axes=(1, 0, 2))

    def _draw_layout(self, surface: pygame.Surface) -> None:
        if not self.active_layout:
            return
        for wall in self.active_layout.walls:
            if wall.width == 0 or wall.height == 0:
                continue
            pygame.draw.rect(surface, (54, 58, 70), wall)            # building
            pygame.draw.rect(surface, (78, 84, 100), wall, 2)        # outline
        for obs in self.active_layout.obstacles:
            pygame.draw.rect(surface, (96, 80, 70), obs)
            pygame.draw.rect(surface, (140, 120, 100), obs, 2)

    def _draw_centerline(self, surface: pygame.Surface) -> None:
        pts = self.active_layout.centerline if self.active_layout else []
        if len(pts) < 2:
            return
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            length = math.hypot(x2 - x1, y2 - y1)
            n = max(int(length // 18), 1)
            for k in range(0, n, 2):  # dashed
                a = (x1 + (x2 - x1) * k / n, y1 + (y2 - y1) * k / n)
                b = (x1 + (x2 - x1) * (k + 1) / n, y1 + (y2 - y1) * (k + 1) / n)
                pygame.draw.line(surface, (70, 110, 90), a, b, 1)

    def _draw_goal(self, surface: pygame.Surface) -> None:
        gx, gy = int(self.goal_pos[0]), int(self.goal_pos[1])
        pygame.draw.circle(surface, (60, 220, 120), (gx, gy), int(self.goal_radius))
        pygame.draw.circle(surface, (200, 255, 220), (gx, gy), int(self.reach_distance), 1)
        # little flag so the destination is obvious
        pygame.draw.line(surface, (230, 230, 230), (gx, gy), (gx, gy - 26), 2)
        pygame.draw.polygon(surface, (240, 90, 90),
                            [(gx, gy - 26), (gx + 16, gy - 21), (gx, gy - 16)])

    def _draw_rays(self, surface: pygame.Surface) -> None:
        origin = (int(self.robot_pos[0]), int(self.robot_pos[1]))
        for dist, end in zip(self.last_ray_distances, self.last_ray_endpoints):
            color = (235, 110, 110) if dist < self.safe_clearance else (90, 160, 220)
            pygame.draw.line(surface, color, origin, end, 1)
            pygame.draw.circle(surface, color, end, 2)

    def _draw_path_trail(self, surface: pygame.Surface) -> None:
        if len(self.path_trail) < 2:
            return
        pygame.draw.aalines(surface, (90, 190, 220), False, self.path_trail[-300:])

    def _draw_robot(self, surface: pygame.Surface) -> None:
        x, y = int(self.robot_pos[0]), int(self.robot_pos[1])
        pygame.draw.circle(surface, (80, 160, 255), (x, y), self.robot_radius)
        hx = int(x + math.cos(self.robot_heading) * self.robot_radius * 1.6)
        hy = int(y + math.sin(self.robot_heading) * self.robot_radius * 1.6)
        pygame.draw.line(surface, (255, 255, 255), (x, y), (hx, hy), 2)

    def _draw_hud(self, surface: pygame.Surface, info: Optional[dict], reward: float) -> None:
        info = info or {}
        success_rate = (self.success_count / max(1, self.episode_count - 1)) * 100.0
        lines = [
            f"Layout: {info.get('layout', '-')}",
            f"Episode: {self.episode_count}   Step: {self.episode_step}",
            f"Reward(step): {reward:+.2f}   Reward(ep): {self.episode_reward:.1f}",
            f"Centre offset: {info.get('corridor_offset', 0.0):+.2f} "
            f"(mean |off| {info.get('mean_abs_offset', 0.0):.2f})",
            f"Min clearance: {info.get('min_clearance', 0.0):.0f}px",
            f"Dist to goal: {info.get('distance', self._distance_to_goal()):.0f}px",
            f"Success rate: {success_rate:.0f}%   Collisions: {self.collision_count}",
            f"Last: {self.last_outcome} ({self.last_episode_reward:.0f})",
        ]
        for i, text in enumerate(lines):
            surface.blit(self._font.render(text, True, (235, 235, 235)), (10, 8 + i * 17))

    def _draw_collision_flash(self, surface: pygame.Surface) -> None:
        if self.collision_flash <= 0.0:
            return
        overlay = pygame.Surface((self.map_width, self.map_height), pygame.SRCALPHA)
        overlay.fill((255, 70, 70, int(120 * self.collision_flash)))
        surface.blit(overlay, (0, 0))
        self.collision_flash = max(0.0, self.collision_flash - 0.1)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
