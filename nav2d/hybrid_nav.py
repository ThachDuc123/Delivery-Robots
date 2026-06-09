"""Hybrid navigation engine for apartment_complex (Zero-Shot Deployment).

Pipeline per the spec:
  * A* on an INFLATED occupancy grid -> waypoints down the CENTRE of corridors
    (never hugging walls).
  * RL local tracker: the trained `ms_mixed` policy follows a SHORT look-ahead
    point (~0.7 m) on that centre path. Because the look-ahead always sits in
    safe free space that A* carved, the RL action stays stable -> no weaving / no
    wall crashes.
  * SafetyShield: DYNAMIC obstacles only (pedestrians). Ignores static walls
    entirely. ±35° cone; proportional steer when a person is < react_dist;
    hard brake when < brake_dist.

Used by hybrid_navigation_eval.ipynb (SLAM -> A* -> hybrid drive -> GIF).
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from world2d import World
from grid_planner import GridPlanner
from delivery_planner import DeliveryPlanner

_HERE = os.path.dirname(os.path.abspath(__file__))
ROBOT_R = 0.22
DT = 0.1
MAX_SPEED = 0.9
MAX_TURN = 2.2


# ----------------------------- map helpers -------------------------------- #
def world_from_grid(grid, cell, origin) -> World:
    """Rebuild a World (wall segments) from an occupancy grid (1 = free)."""
    g = grid; ox, oy = origin; R, C = g.shape
    def wall(r, c): return not (0 <= r < R and 0 <= c < C and g[r, c] == 1)
    segs = []
    for r in range(R):
        for c in range(C):
            if g[r, c] != 1:
                continue
            x0 = ox + c*cell; y0 = oy + r*cell; x1 = x0 + cell; y1 = y0 + cell
            if wall(r, c-1): segs.append((x0, y0, x0, y1))
            if wall(r, c+1): segs.append((x1, y0, x1, y1))
            if wall(r-1, c): segs.append((x0, y0, x1, y0))
            if wall(r+1, c): segs.append((x0, y1, x1, y1))
    w = World(half_width=cell*2.0, style="hybrid")
    w.segments = segs
    w.bounds = (ox - cell, ox + C*cell + cell, oy - cell, oy + R*cell + cell)
    return w


def inflate_map(grid: np.ndarray, layers: int = 1) -> np.ndarray:
    """Shrink free space by `layers` cells around every wall (inflation layer):
    A* on the result keeps clearance, so paths run down the corridor centre."""
    g = grid.copy()
    R, C = g.shape
    for _ in range(layers):
        free = g == 1
        keep = free.copy()
        for r in range(R):
            for c in range(C):
                if not free[r, c]:
                    continue
                for dr, dc in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                    rr, cc = r+dr, c+dc
                    if not (0 <= rr < R and 0 <= cc < C and free[rr, cc]):
                        keep[r, c] = False; break
        g = keep.astype(np.uint8)
    return g


# --------------------- A* centre-hugging waypoints ------------------------ #
def plan_route(grid, cell, origin, dock, points, order, inflate_layers=1):
    """Return per-leg centre-path waypoint lists for dock->order...->dock."""
    base = {"grid": grid, "cell": cell, "origin": origin}
    infl = {"grid": inflate_map(grid, inflate_layers), "cell": cell, "origin": origin}
    p_in = GridPlanner(infl, inflate=0)        # already-inflated grid
    p_raw = GridPlanner(base, inflate=0)       # fallback for niches the inflation closed
    legs = []; prev = tuple(dock)
    for s in list(order) + ["dock"]:
        tgt = tuple(dock) if s == "dock" else points[s]
        wps, _ = p_in.plan(prev, tgt)
        if wps is None:
            wps, _ = p_raw.plan(prev, tgt)
        legs.append(wps); prev = tgt
    return legs


def tsp_order(grid, cell, origin, dock, points, picks):
    dp = DeliveryPlanner({"world": None, "dock": tuple(dock), "points": points,
                          "grid": grid, "cell": cell, "origin": origin})
    dp.planner = GridPlanner({"grid": grid, "cell": cell, "origin": origin}, inflate=0)
    return dp.optimize(list(picks))["order"]


# --------------------------- Safety Shield (dynamic only) ----------------- #
class SafetyShield:
    """Pedestrian-only avoidance. Ignores walls. Proportional, smooth."""
    def __init__(self, react=1.0, brake=0.5, cone_deg=35.0):
        self.react = react; self.brake = brake; self.cone = math.radians(cone_deg)

    def adjust(self, pos, heading, peds):
        """Return (speed_scale, extra_turn, status). extra_turn ∝ closeness."""
        if peds is None or len(peds.pos) == 0:
            return 1.0, 0.0, "clear"
        fwd = np.array([math.cos(heading), math.sin(heading)])
        left = np.array([-fwd[1], fwd[0]])
        worst = None
        for p in peds.pos:
            rel = np.array(p) - pos; d = float(np.linalg.norm(rel))
            if d > self.react:
                continue
            ang = abs(math.atan2(rel @ left, rel @ fwd))
            if (rel @ fwd) > 0 and ang < self.cone and (worst is None or d < worst[0]):
                worst = (d, float(rel @ left))
        if worst is None:
            return 1.0, 0.0, "clear"
        d, side = worst
        closeness = 1.0 - (d - self.brake) / max(self.react - self.brake, 1e-6)  # 0..1
        closeness = float(np.clip(closeness, 0.0, 1.0))
        extra_turn = -np.sign(side or 1.0) * closeness * 0.9      # steer away, ∝ closeness
        if d < self.brake:
            return 0.0, extra_turn, "brake"
        return (1.0 - 0.6 * closeness), extra_turn, "slow_sidestep"


# --------------------------- RL local tracker ----------------------------- #
class RLTracker:
    """Wrap ms_mixed: feed it the env obs aimed at a 0.7 m look-ahead point."""
    def __init__(self, model_name="ms_mixed", lookahead=0.7):
        from sb3_contrib import RecurrentPPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from multistop_env import MultiStopEnv
        self.model = RecurrentPPO.load(os.path.join(_HERE, "runs", f"{model_name}.zip"))
        vec = VecNormalize.load(os.path.join(_HERE, "runs", f"{model_name}_vecnorm.pkl"),
                                DummyVecEnv([lambda: MultiStopEnv(config=dict(procedural=True))]))
        self.mean = vec.obs_rms.mean.astype(np.float32); self.var = vec.obs_rms.var.astype(np.float32)
        self.eps = vec.epsilon; self.clip = vec.clip_obs
        self.lookahead = lookahead
        self._env = MultiStopEnv(config=dict(n_lidar=24, lidar_range=5.0))
        self.state = None; self.es = np.ones(1, bool)

    def reset(self):
        self.state = None; self.es = np.ones(1, bool)

    def _norm(self, o):
        return np.clip((o - self.mean) / np.sqrt(self.var + self.eps), -self.clip, self.clip).astype(np.float32)

    def lookahead_point(self, pos, path, wi):
        while wi < len(path) - 1 and np.linalg.norm(np.array(path[wi]) - pos) < self.lookahead:
            wi += 1
        return np.array(path[wi]), wi

    def action(self, world, pos, heading, look):
        """RL action [forward(-..1), turn(-1..1)] tracking the look-ahead point."""
        e = self._env; e.world = world
        e.pos = np.array(pos, float); e.heading = heading
        e.path = [tuple(pos), tuple(look)]; e.seg_i = 0
        e.legs = [e.path]; e.leg_i = 0; e.stops_total = 1; e.stops_done = 0
        e.grace = 0; e.prev_turn = 0.0
        o = e._obs()
        a, self.state = self.model.predict(self._norm(o)[None], state=self.state,
                                           episode_start=self.es, deterministic=True)
        self.es = np.zeros(1, bool)
        return a[0]


# ----------------- Guided-RL tracker (ms_guided_smooth) ------------------- #
class GuidedTracker:
    """Bộ lái dùng model Guided-RL (ms_guided_smooth, obs 28-chiều ego-relative
    [24 LiDAR, e_y, e_theta, v, omega]). Dùng làm local_policy cho MissionController:
    - on_leg(path): nhận tuyến A* của chặng (để tính e_y/e_theta).
    - __call__(world,pos,heading,look) -> (fwd_cmd in[-1,1], turn rad/s).
    """
    V_MAX = MAX_SPEED; W_MAX = MAX_TURN; N = 24; RANGE = 5.0

    def __init__(self, model_name="ms_guided_smooth"):
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from track_env import TrackEnv
        self.model = PPO.load(os.path.join(_HERE, "runs", f"{model_name}.zip"))
        vec = VecNormalize.load(os.path.join(_HERE, "runs", f"{model_name}_vecnorm.pkl"),
                                DummyVecEnv([lambda: TrackEnv()]))
        self.mean = vec.obs_rms.mean.astype(np.float32); self.var = vec.obs_rms.var.astype(np.float32)
        self.eps = vec.epsilon; self.clip = vec.clip_obs
        self.angles = np.linspace(0, 2*math.pi, self.N, endpoint=False)
        self.path = None; self.seg_i = 0; self.v = 0.0; self.w = 0.0

    def on_leg(self, path):
        self.path = [np.array(p, float) for p in path]
        self.seg_i = 0; self.v = 0.0; self.w = 0.0

    def _nearest(self, pos):
        P = self.path; best = (1e9, 0.0, 0.0, self.seg_i)
        lo = max(0, self.seg_i - 2); hi = min(len(P) - 1, self.seg_i + 8)
        for i in range(lo, hi):
            a = P[i]; b = P[i+1]; ab = b - a; L2 = float(ab @ ab) or 1e-9
            t = float(np.clip(((pos - a) @ ab) / L2, 0, 1)); proj = a + t * ab
            dist = float(np.linalg.norm(pos - proj))
            if dist < best[0]:
                tang = math.atan2(ab[1], ab[0])
                left = np.array([-math.sin(tang), math.cos(tang)])
                best = (dist, float((pos - proj) @ left), tang, i)
        return best

    def __call__(self, world, pos, heading, look):
        pos = np.array(pos, float)
        if self.path is None:
            self.path = [pos, np.array(look, float)]
        dist, ey, tang, idx = self._nearest(pos); self.seg_i = idx
        etheta = (tang - heading + math.pi) % (2*math.pi) - math.pi
        ang = heading + self.angles
        lidar = world.raycast_batch(tuple(pos), ang, self.RANGE) / self.RANGE
        obs = np.concatenate([
            np.clip(lidar, 0, 1),
            [float(np.clip(ey / 1.0, -1, 1))],
            [float(etheta / math.pi)],
            [float(np.clip(self.v / self.V_MAX, -1, 1))],
            [float(np.clip(self.w / self.W_MAX, -1, 1))],
        ]).astype(np.float32)
        obs = np.clip(obs, -1, 1)
        o = np.clip((obs - self.mean) / np.sqrt(self.var + self.eps), -self.clip, self.clip).astype(np.float32)
        a, _ = self.model.predict(o[None], deterministic=True); a = a[0]
        self.v = (float(a[0]) * 0.7 + 0.3) * self.V_MAX
        self.w = float(a[1]) * self.W_MAX
        return float(a[0]), self.w


# --------------------------- full drive ----------------------------------- #
def run_delivery(grid, cell, origin, dock, points, picks, peds=None,
                 inflate_layers=1, max_steps=12000, lookahead=0.7, rl_weight=0.3):
    world = world_from_grid(grid, cell, origin)
    order = tsp_order(grid, cell, origin, dock, points, picks)
    legs = plan_route(grid, cell, origin, dock, points, order, inflate_layers)
    # densify legs for smooth tracking
    legs = [_densify(l, 0.2) for l in legs]
    full_route = [p for leg in legs for p in leg]

    tracker = RLTracker(lookahead=lookahead); tracker.reset()
    shield = SafetyShield()
    pos = np.array(dock, float); heading = 0.0
    trail = [tuple(pos)]; pedtrail = [[tuple(p) for p in peds.pos]] if peds else []
    avoid_pts = []
    targets = [points[s] for s in order] + [tuple(dock)]
    labels = [str(s) for s in order] + ["dock"]
    delivered = []; steps = 0; ped_hits = 0
    turn_s = 0.0                       # smoothed turn command (low-pass -> no wobble)
    TURN_ALPHA = 0.2                   # smoothing factor
    TURN_RATE = 1.6                    # max |Δheading| per step (rad/s * dt clamp)
    last_pos = pos.copy(); stall = 0   # recovery state for tight corners

    for leg, tgt, lab in zip(legs, targets, labels):
        wi = 1; guard = 0
        while np.linalg.norm(np.array(tgt) - pos) > 0.4 and steps < max_steps and guard < 3000:
            guard += 1; steps += 1
            # --- corner recovery: only when REALLY stuck for a while, fan headings
            #     toward the next path cell to round a tight corner, then resume.
            if float(np.linalg.norm(pos - last_pos)) < 0.01:
                stall += 1
            else:
                stall = max(0, stall - 2)
            last_pos = pos.copy()
            if stall >= 30:
                nxt = np.array(leg[min(wi + 2, len(leg) - 1)])
                base = math.atan2(nxt[1] - pos[1], nxt[0] - pos[0])
                for off in (0.0, 0.6, -0.6, 1.2, -1.2, math.pi):
                    cand = pos + np.array([math.cos(base+off), math.sin(base+off)]) * (MAX_SPEED*0.5*DT)
                    if not world.segment_hits_circle(tuple(cand), ROBOT_R) and \
                       not (peds is not None and peds.hits_robot(tuple(cand), ROBOT_R)):
                        pos = cand; heading = (base+off+math.pi)%(2*math.pi)-math.pi
                        turn_s = 0.0; stall = 0; wi = min(wi+1, len(leg)-1); break
                trail.append(tuple(pos))
                if peds: pedtrail.append([tuple(p) for p in peds.pos])
                continue
            if peds: peds.step(DT)
            look, wi = tracker.lookahead_point(pos, leg, wi)
            a = tracker.action(world, pos, heading, look)
            fwd = (float(a[0]) * 0.7 + 0.3) * MAX_SPEED
            rl_turn = float(a[1]) * MAX_TURN
            # pure-pursuit turn toward the look-ahead (smooth centre-line backbone)
            dxy = np.array(look) - pos
            bearing = (math.atan2(dxy[1], dxy[0]) - heading + math.pi) % (2*math.pi) - math.pi
            pp_turn = float(np.clip(2.5 * bearing, -MAX_TURN, MAX_TURN))  # proportional gain
            # blend: pursuit dominant (straight, hugs centre) + RL contribution
            turn = (1.0 - rl_weight) * pp_turn + rl_weight * rl_turn
            # dynamic-only safety shield
            sscale, sturn, status = shield.adjust(pos, heading, peds)
            if status != "clear":
                avoid_pts.append(tuple(pos))
            fwd *= sscale; turn += sturn * MAX_TURN
            # low-pass + rate-limit the turn so heading changes are smooth (no
            # frame-to-frame flip -> straight, non-weaving trajectory)
            turn_s += TURN_ALPHA * (turn - turn_s)
            dh = float(np.clip(turn_s * DT, -TURN_RATE * DT, TURN_RATE * DT))
            heading = (heading + dh + math.pi) % (2*math.pi) - math.pi
            npos = pos + np.array([math.cos(heading), math.sin(heading)]) * fwd * DT
            if not world.segment_hits_circle(tuple(npos), ROBOT_R):
                if peds is not None and peds.hits_robot(tuple(npos), ROBOT_R):
                    ped_hits += 1
                else:
                    pos = npos
            else:
                wi = min(wi + 1, len(leg) - 1)   # nudge past a corner
            trail.append(tuple(pos))
            if peds: pedtrail.append([tuple(p) for p in peds.pos])
        if lab != "dock":
            delivered.append(int(lab))

    back = float(np.linalg.norm(pos - np.array(dock)))
    return {"order": [int(x) for x in order], "delivered": delivered,
            "returned_dock": back <= 0.6, "steps": steps, "ped_hits": ped_hits,
            "trail": trail, "pedtrail": pedtrail, "avoid_pts": avoid_pts,
            "route": full_route, "world": world, "picks": list(picks)}


def _densify(path, gap=0.2):
    if not path or len(path) < 2:
        return [tuple(p) for p in (path or [])]
    out = [tuple(path[0])]
    for a, b in zip(path[:-1], path[1:]):
        a = np.array(a, float); b = np.array(b, float)
        d = float(np.linalg.norm(b - a)); n = max(1, int(math.ceil(d / gap)))
        for k in range(1, n + 1):
            out.append(tuple(a + (b - a) * k / n))
    return out
