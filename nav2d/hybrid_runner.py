"""Hybrid multi-stop delivery runner: Global Planner + RL/LiDAR local control.

Architecture (matches the requested design):
  * **Global Planner** (grid A* + TSP, `delivery_planner`): from the dock and the
    chosen delivery points, compute the optimal visiting ORDER and per-leg routes;
    deliver ALL points, then return to the dock (battery-friendly single trip).
  * **Local controller** = the PPO policy trained on `DeliveryFollowEnv`
    (`runs/ppo_delivery`): drives each leg using only LiDAR + a pure-pursuit
    lookahead point on the planned route, so it follows the line smoothly and
    avoids walls. This is the policy that fixed the earlier weaving.

We reuse ``DeliveryFollowEnv`` itself as the execution engine for one leg at a
time (its observation, lookahead and dynamics are exactly what the policy was
trained on — no train/deploy mismatch), overriding the leg's route so the robot
goes dock -> p1 -> p2 -> p3 -> dock.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

from fixed_maps import build_map
from delivery_planner import DeliveryPlanner
from delivery_train_env import DeliveryFollowEnv
from grid_planner import GridPlanner

_HERE = os.path.dirname(os.path.abspath(__file__))
FRAME_STACK = 4


class HybridDeliveryRunner:
    def __init__(self, map_name: str = "apartment_a",
                 model_path: Optional[str] = None, vec_path: Optional[str] = None,
                 max_steps_per_leg: int = 900):
        self.map = build_map(map_name)
        self.world = self.map["world"]
        self.map_name = map_name
        self.dp = DeliveryPlanner(self.map)
        self._plan_inflated = GridPlanner(self.map, inflate=1)
        self._plan_plain = GridPlanner(self.map, inflate=0)
        self.max_steps_per_leg = max_steps_per_leg

        model_path = model_path or os.path.join(_HERE, "runs", "ppo_delivery")
        vec_path = vec_path or os.path.join(_HERE, "runs", "ppo_delivery_vecnorm.pkl")
        self.model = PPO.load(model_path)
        venv = VecFrameStack(DummyVecEnv([lambda: DeliveryFollowEnv()]), FRAME_STACK)
        venv = VecNormalize.load(vec_path, venv)
        self._mean = venv.obs_rms.mean.astype(np.float32)
        self._var = venv.obs_rms.var.astype(np.float32)
        self._eps = venv.epsilon; self._clip = venv.clip_obs

        # one engine env bound to this map (we drive it leg-by-leg)
        self.env = DeliveryFollowEnv(config=dict(n_lidar=24, lidar_range=5.0,
                                                 max_steps=max_steps_per_leg, lookahead=1.6,
                                                 maps=[map_name]))
        self.env.reset(seed=0)
        self.arrive_dist = self.env.arrive_dist

    def _norm(self, o):
        return np.clip((o - self._mean) / np.sqrt(self._var + self._eps),
                       -self._clip, self._clip).astype(np.float32)

    def _plan_leg(self, a, b):
        wps, L = self._plan_inflated.plan(a, b)
        if wps is None:
            wps, L = self._plan_plain.plan(a, b)
        return wps, L

    def _drive_leg(self, start, goal, trail, stack):
        """Drive one leg start->goal with the trained policy; returns (end_pos,
        arrived, collisions, steps, stack)."""
        e = self.env
        wps, _ = self._plan_leg(start, goal)
        if wps is None:
            return tuple(start), False, 0, 0, stack
        # set the engine env onto this leg's route
        e.world = self.world
        e.path = list(wps); e.path_len = sum(
            math.hypot(wps[i+1][0]-wps[i][0], wps[i+1][1]-wps[i][1]) for i in range(len(wps)-1))
        e.pos = np.array(start, dtype=np.float64)
        e.goal = np.array(goal, dtype=np.float64)
        d = np.array(wps[1]) - np.array(wps[0]) if len(wps) > 1 else np.array([1., 0.])
        e.heading = math.atan2(d[1], d[0])
        e.prev_turn = 0.0; e.seg_i = 0; e.bump = 0; e.step_i = 0
        e.prev_along = 0.0

        coll = 0; steps = 0; arrived = False
        o = e._obs()
        while steps < self.max_steps_per_leg:
            stack = np.tile(o, FRAME_STACK) if stack is None else np.concatenate([stack[len(o):], o])
            a, _ = self.model.predict(self._norm(stack), deterministic=True)
            o, r, term, trunc, info = e.step(a)
            trail.append(tuple(e.pos)); steps += 1
            coll += int(info["collision"])
            if info["arrived"]:
                arrived = True; break
            if term or trunc:
                break
        return tuple(e.pos), arrived, coll, steps, stack

    def run(self, chosen_ids: List[int], seed: int = 0) -> Dict:
        plan = self.dp.optimize(chosen_ids)
        if not plan["reachable"]:
            return {"plan": plan, "success": False, "reason": "unreachable", "trail": []}

        order = plan["order"]
        targets = [self.map["points"][p] for p in order] + [self.map["dock"]]
        labels = [str(p) for p in order] + ["dock"]
        pos = tuple(self.map["dock"]); trail = [pos]; stack = None
        delivered = []; collisions = 0; steps = 0
        for tgt, lab in zip(targets, labels):
            pos, arrived, c, s, stack = self._drive_leg(pos, tgt, trail, stack)
            collisions += c; steps += s
            if lab != "dock" and arrived:
                delivered.append(int(lab))
            elif lab != "dock" and not arrived:
                # failed to reach this stop -> abort remaining (report partial)
                break
        back = math.hypot(pos[0] - self.map["dock"][0], pos[1] - self.map["dock"][1])
        returned = back <= max(self.arrive_dist * 1.5, 0.7)
        success = (len(delivered) == len(order)) and returned
        return {"plan": plan, "success": success, "delivered": delivered, "order": order,
                "trail": trail, "steps": steps, "collisions": collisions,
                "sim_time_s": steps * self.env.dt, "battery_pct": plan["battery_pct"],
                "total_dist": plan["total_dist"], "returned_dock": returned}
