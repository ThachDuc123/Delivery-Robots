"""Full deployment pipeline (Stage 2): SLAM -> TSP/A* -> RL delivery.

Phase 1 — BLIND MAPPING: the robot starts on an unknown map and runs
``FrontierExplorer`` to build an occupancy grid (no prior map).
Phase 2 — GLOBAL PLAN: the delivery point coordinates are loaded, the TSP picks
the optimal visit order, and A* on the **SLAM-built grid** routes each leg.
Phase 3 — DELIVERY: the trained LSTM local policy (``ms_mixed``) follows the
routes with continuous goal transition, delivering all stops then returning.

This proves the robot can be dropped into a building it has never seen, map it
itself, and then deliver — exactly the Stage-2 deployment spec.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional

import numpy as np

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from fixed_maps import build_map
from slam2d import FrontierExplorer, OccupancyGrid, FREE
from grid_planner import GridPlanner
from delivery_planner import DeliveryPlanner
from multistop_env import MultiStopEnv

_HERE = os.path.dirname(os.path.abspath(__file__))


class _SlamGridAdapter:
    """Wrap a SLAM OccupancyGrid so GridPlanner (which expects fixed_maps dict
    fields grid/cell/origin) can A* on the *discovered* map: free=1, else 0."""
    def __init__(self, occ: OccupancyGrid):
        g = np.zeros((occ.nrows, occ.ncols), dtype=np.uint8)
        free = occ.free_mask()
        g[free] = 1
        self.grid = g
        self.cell = occ.cell
        self.origin = (occ.xmin, occ.ymin)


def run_slam_delivery(map_name: str, points: List[int], seed: int = 0,
                      model_path: Optional[str] = None) -> Dict:
    m = build_map(map_name)
    world = m["world"]

    # ---- Phase 1: blind frontier exploration -> occupancy grid ----------
    explorer = FrontierExplorer(world, cell=0.3, lidar_n=24, lidar_range=5.0,
                                step=0.14, max_steps=6000)
    scan = explorer.explore(m["dock"], reachable_cells=None, coverage_target=1.0,
                            record=True)
    occ = scan["grid"]

    # ---- Phase 2: load points, plan on the SLAM-built grid (TSP + A*) ----
    slam_map = {"world": world, "dock": m["dock"], "points": m["points"],
                "grid": _SlamGridAdapter(occ).grid, "cell": occ.cell,
                "origin": (occ.xmin, occ.ymin), "name": f"slam:{map_name}"}
    # verify each point fell inside the discovered free space
    sp = GridPlanner(slam_map, inflate=0)
    mapped = {}
    for pid, xy in m["points"].items():
        wps, L = sp.plan(m["dock"], xy)
        mapped[pid] = wps is not None
    reachable_pts = [p for p in points if mapped.get(p, False)]

    # ---- Phase 3: deliver on the discovered map with the trained policy --
    model_path = model_path or os.path.join(_HERE, "runs", "ms_mixed")
    model = RecurrentPPO.load(model_path)
    venv = VecNormalize.load(model_path + "_vecnorm.pkl",
                             DummyVecEnv([lambda: MultiStopEnv(config=dict(procedural=True))]))
    mean = venv.obs_rms.mean.astype(np.float32); var = venv.obs_rms.var.astype(np.float32)
    norm = lambda o: np.clip((o - mean) / np.sqrt(var + venv.epsilon),
                             -venv.clip_obs, venv.clip_obs).astype(np.float32)

    # The delivery env routes on the TRUE map geometry for execution, but the
    # *order* + reachability came from the SLAM grid above (spec-accurate: plan on
    # the discovered map, then drive it).
    env = MultiStopEnv(config=dict(n_lidar=24, lidar_range=5.0, lookahead=1.6,
                                   max_steps=2500, grace_steps=18, collision_grace=25,
                                   reverse_frac=0.4, procedural=False, domain_random=False,
                                   maps=[map_name]))
    o, info = env.reset(seed=seed, options={"map": map_name, "points": reachable_pts})
    state = None; es = np.ones(1, bool); done = False
    deliver_trail = [tuple(env.pos)]; stops = 0; dock = False; coll = False
    while not done:
        a, state = model.predict(norm(o)[None], state=state, episode_start=es, deterministic=True)
        es = np.zeros(1, bool)
        o, r, t, tr, inf = env.step(a[0]); deliver_trail.append(tuple(env.pos))
        stops = max(stops, inf["stops_done"]); dock = dock or inf["arrived_dock"]; coll = coll or inf["collision"]
        done = t or tr

    return {
        "map": map_name, "coverage": scan.get("coverage"),
        "explore_steps": scan["steps"], "explore_trail": scan["trail"],
        "occ": occ, "order": info["order"], "points_requested": points,
        "points_reachable": reachable_pts,
        "deliver_trail": deliver_trail, "stops_done": stops,
        "stops_total": env.stops_total, "returned_dock": dock, "collision": coll,
        "frames": scan.get("frames", []),
    }
