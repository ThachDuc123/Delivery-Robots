"""track_env.py — Guided-RL path-tracking environment (chống học vẹt map).

Triết lý:
  * STATE ego-relative, MÙ về map (không có toạ độ tuyệt đối X,Y, không tên map):
      [ 24 tia LiDAR (chỉ tường) | e_y (lệch tim đường) | e_theta (lệch hướng) |
        v_t | omega_t ]
    -> thả vào 1000 map khác nhau, trạng thái vẫn đúng -> KHÔNG thể học vẹt map.
    (Việc "nhớ map" là của SLAM/A*, không phải của mạng lái.)
  * GUIDED RL reward (giữ đúng tính cách PPO, vừa chạy vừa học):
      R = R_nav + R_collision - K * (omega_RL - omega_PP)^2
    omega_PP = góc lái lý tưởng của Pure-Pursuit tính tại CÙNG vị trí. RL lái càng
    giống Pure-Pursuit càng ít bị trừ; NHƯNG khi LiDAR báo sắp đâm tường, phạt va
    chạm (nặng) lớn hơn phạt lệch PP -> RL tự "phá khuôn" để né, qua rồi ôm lại
    đường A*. -> lấy toán học (PP) làm kim chỉ nam, dùng cảm biến để sinh tồn.

Env sinh 1 tuyến A* (dock -> 1 điểm) trên map ngẫu nhiên mỗi episode; nhiệm vụ là
BÁM tuyến đó (sợi dây A) cho mượt như Pure-Pursuit.
"""

from __future__ import annotations

import glob
import json
import math
import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from world2d import World
from hybrid_controller import AStarPlanner, inflate_map

_HERE = os.path.dirname(os.path.abspath(__file__))
MAP_DIR = os.path.join(_HERE, "data", "maps")

V_MAX = 0.9
W_MAX = 2.2
DT = 0.1
ROBOT_R = 0.22
N_LIDAR = 24
LIDAR_RANGE = 5.0
LOOKAHEAD = 0.7         # pure-pursuit lookahead (m)


def _world_from_grid(grid, cell, origin):
    ox, oy = origin; R, C = grid.shape
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
    w = World(half_width=cell*2.0, style="track")
    w.segments = segs
    w.bounds = (ox - cell, ox + C*cell + cell, oy - cell, oy + R*cell + cell)
    return w


def _densify(path, gap=0.15):
    if not path or len(path) < 2:
        return [tuple(p) for p in (path or [])]
    out = [tuple(path[0])]
    for a, b in zip(path[:-1], path[1:]):
        a = np.array(a, float); b = np.array(b, float)
        d = float(np.linalg.norm(b - a)); n = max(1, int(math.ceil(d / gap)))
        for k in range(1, n + 1):
            out.append(tuple(a + (b - a) * k / n))
    return out


class TrackEnv(gym.Env):
    def __init__(self, config=None):
        c = config or {}
        self.map_dir = c.get("map_dir", MAP_DIR)
        self.max_steps = int(c.get("max_steps", 1500))
        self.domain_random = bool(c.get("domain_random", True))
        self.K = float(c.get("K_guide", 0.5))          # trọng số bắt chước Pure-Pursuit
        self.w_dw = float(c.get("w_dw", 0.0))          # phạt tốc độ ĐỔI LÁI |Δω| -> diệt đánh võng
        self.w_omega = float(c.get("w_omega", 0.0))    # phạt nhẹ độ lớn |ω|
        self.w_prog = float(c.get("w_prog", 2.0))
        self.w_time = float(c.get("w_time", 0.01))
        self.w_collide = float(c.get("w_collide", 4.0))
        self.w_jam = float(c.get("w_jam", 60.0))
        self.collision_grace = int(c.get("collision_grace", 20))
        self.w_arrive = float(c.get("w_arrive", 20.0))
        # nguồn map: hoặc danh sách map dict cố định (config["maps"]) để eval map cũ,
        # hoặc index.json của data/maps (mặc định).
        self.fixed_maps = c.get("maps", None)
        if self.fixed_maps is None:
            with open(os.path.join(self.map_dir, "index.json"), encoding="utf-8") as f:
                self._index = json.load(f)
        self.lidar_angles = np.linspace(0, 2*math.pi, N_LIDAR, endpoint=False)
        # obs = 24 lidar + e_y + e_theta + v + omega = 28
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(N_LIDAR + 4,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.rng = np.random.default_rng(0)

    # ----------------------------- reset ------------------------------- #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # sample map + a valid A* path (dock -> a reachable point)
        for _ in range(20):
            if self.fixed_maps is not None:
                m = self.fixed_maps[self.rng.integers(len(self.fixed_maps))]
                grid = m["grid"].astype(np.uint8); cell = float(m["cell"])
                origin = tuple(m["origin"]); dock = tuple(m["dock"])
                pts = {int(k): tuple(v) for k, v in m["points"].items()}
            else:
                e = self._index[self.rng.integers(len(self._index))]
                grid = np.load(os.path.join(self.map_dir, e["file"])).astype(np.uint8)
                cell = float(e["cell"]); origin = tuple(e["origin"])
                dock = tuple(e["dock"]); pts = {int(k): tuple(v) for k, v in e["points"].items()}
            if not pts:
                continue
            world = _world_from_grid(grid, cell, origin)
            planner = AStarPlanner(grid, cell, origin, inflate_layers=1)
            pid = list(pts)[self.rng.integers(len(pts))]
            # đôi khi đi ngược (point -> dock) để đa dạng hướng
            a, b = (dock, pts[pid]) if self.rng.random() < 0.5 else (pts[pid], dock)
            wp = planner.plan(a, b)
            if wp and len(wp) >= 2:
                self.world = world; self.path = _densify(wp, 0.15)
                break
        else:
            # fallback: tuyến thẳng ngắn
            self.world = world; self.path = _densify([dock, pts[list(pts)[0]]], 0.15)

        self._speed_mult = 1.0; self._turn_mult = 1.0; self._lidar_noise = 0.0
        if self.domain_random:
            self._speed_mult = float(self.rng.uniform(0.9, 1.1))
            self._turn_mult = float(self.rng.uniform(0.9, 1.1))
            self._lidar_noise = float(self.rng.uniform(0.0, 0.02))

        self.pos = np.array(self.path[0], float)
        d0 = np.array(self.path[1]) - np.array(self.path[0])
        self.heading = math.atan2(d0[1], d0[0])
        self.v = 0.0; self.w = 0.0; self.prev_w = 0.0
        self.seg_i = 0; self.step_i = 0; self.bump = 0
        self.prev_along = 0.0
        return self._obs(), {}

    # --------------------- path geometry helpers ----------------------- #
    def _nearest_seg(self):
        """Tìm đoạn path gần nhất -> trả (cross-track signed, tangent_heading, idx)."""
        P = self.path; pos = self.pos
        best = (1e9, 0.0, 0.0, self.seg_i)
        lo = max(0, self.seg_i - 2); hi = min(len(P) - 1, self.seg_i + 8)
        for i in range(lo, hi):
            a = np.array(P[i]); b = np.array(P[i+1]); ab = b - a
            L2 = float(ab @ ab) or 1e-9
            t = float(np.clip(((pos - a) @ ab) / L2, 0, 1))
            proj = a + t * ab; dist = float(np.linalg.norm(pos - proj))
            if dist < best[0]:
                tang = math.atan2(ab[1], ab[0])
                # dấu cross-track: bên trái(+)/phải(-) của hướng tuyến
                left = np.array([-math.sin(tang), math.cos(tang)])
                signed = float((pos - proj) @ left)
                best = (dist, signed, tang, i)
        return best  # (dist, signed_ey, tangent_heading, seg_idx)

    def _lookahead_point(self):
        """Điểm cách mũi xe ~LOOKAHEAD dọc path (cho Pure-Pursuit)."""
        P = self.path; acc = 0.0; prev = self.pos.copy()
        for j in range(self.seg_i, len(P)):
            q = np.array(P[j]); d = float(np.linalg.norm(q - prev))
            if acc + d >= LOOKAHEAD:
                t = (LOOKAHEAD - acc) / max(d, 1e-6)
                return prev + (q - prev) * t
            acc += d; prev = q
        return np.array(P[-1])

    def _omega_pp(self):
        """Góc lái lý tưởng của Pure-Pursuit tại vị trí hiện tại."""
        look = self._lookahead_point()
        bearing = (math.atan2(look[1]-self.pos[1], look[0]-self.pos[0]) - self.heading + math.pi) % (2*math.pi) - math.pi
        return float(np.clip(2.2 * bearing, -W_MAX, W_MAX))

    def _lidar(self):
        ang = self.heading + self.lidar_angles
        r = self.world.raycast_batch(tuple(self.pos), ang, LIDAR_RANGE) / LIDAR_RANGE
        if self._lidar_noise > 0:
            r = r + self.rng.normal(0, self._lidar_noise, size=r.shape)
        return np.clip(r, 0.0, 1.0).astype(np.float32)

    # ------------------------------ obs -------------------------------- #
    def _obs(self):
        dist, ey, tang, idx = self._nearest_seg()
        self.seg_i = idx
        etheta = (tang - self.heading + math.pi) % (2*math.pi) - math.pi
        lidar = self._lidar()
        obs = np.concatenate([
            lidar,                                  # 24: tôi cách tường bao xa
            [float(np.clip(ey / 1.0, -1, 1))],      # e_y: lệch tim đường (m, ±1)
            [float(etheta / math.pi)],              # e_theta: lệch hướng (±1)
            [float(np.clip(self.v / V_MAX, -1, 1))],# vận tốc hiện tại
            [float(np.clip(self.w / W_MAX, -1, 1))],# tốc độ xoay hiện tại
        ]).astype(np.float32)
        return np.clip(obs, -1.0, 1.0)

    def _along(self):
        """Quãng đường đã đi dọc path (để tính progress)."""
        P = self.path; s = 0.0
        for i in range(self.seg_i):
            s += float(np.linalg.norm(np.array(P[i+1]) - np.array(P[i])))
        a = np.array(P[self.seg_i]); b = np.array(P[min(self.seg_i+1, len(P)-1)])
        ab = b - a; L2 = float(ab @ ab) or 1e-9
        t = float(np.clip(((self.pos - a) @ ab) / L2, 0, 1))
        return s + t * float(np.linalg.norm(ab))

    # ------------------------------ step ------------------------------- #
    def step(self, action):
        self.step_i += 1
        a = np.clip(np.asarray(action, np.float32), -1, 1)
        v = (0.7 * a[0] + 0.3) * V_MAX
        v = float(np.clip(v, -0.4 * V_MAX, V_MAX))
        w = float(a[1]) * W_MAX
        self.v, self.w = v, w
        omega_pp = self._omega_pp()                 # tính TRƯỚC khi di chuyển (cùng vị trí)

        nh = (self.heading + w * self._turn_mult * DT + math.pi) % (2*math.pi) - math.pi
        npos = self.pos + np.array([math.cos(nh), math.sin(nh)]) * v * self._speed_mult * DT
        collided = self.world.segment_hits_circle(tuple(npos), ROBOT_R)
        if not collided:
            self.pos = npos; self.bump = 0
        else:
            self.bump += 1
        self.heading = nh

        # ---- reward: R_nav + R_collision - K*(w_RL - w_PP)^2 ----
        along = self._along()
        reward = self.w_prog * (along - self.prev_along)     # R_nav (tiến dọc tuyến)
        self.prev_along = along
        reward -= self.w_time
        reward -= self.K * (w - omega_pp) ** 2               # "hồn Pure-Pursuit"
        reward -= self.w_dw * abs(w - self.prev_w)           # phạt đổi lái đột ngột -> mượt
        reward -= self.w_omega * abs(w)                      # phạt lái gắt
        self.prev_w = w

        terminated = False
        info = {"collision": False, "omega_pp": omega_pp, "omega_rl": w}
        if collided:
            reward -= self.w_collide                          # R_collision
            if self.bump > self.collision_grace:
                reward -= self.w_jam; terminated = True; info["collision"] = True

        # tới cuối tuyến?
        if not terminated and float(np.linalg.norm(self.pos - np.array(self.path[-1]))) <= 0.35:
            reward += self.w_arrive; terminated = True; info["arrived"] = True

        truncated = self.step_i >= self.max_steps
        return self._obs(), float(reward), terminated, truncated, info
