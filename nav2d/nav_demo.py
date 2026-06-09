"""End-to-end delivery demo engine on apartment_complex_v1, in 3 phases.

Phase 1  build_and_save_map()  -- robot starts BLIND, frontier-explores to build
          an occupancy grid (SLAM), and saves it to disk (npz).
Phase 2  deliver(picks)        -- load the saved map, TSP-order the chosen
          delivery points, A* route each leg on the saved map, and drive with a
          smooth pure-pursuit controller (deliver all, return to dock).
Phase 3  deliver(picks, peds)  -- same, but with moving residents; a LiDAR safety
          shield detects a person ahead (small reaction radius) and nudges the
          robot smoothly aside, then eases back onto the path (no weaving).

Pure geometry + the existing SLAM/planner — no RL policy needed for the demo, so
the motion is fully deterministic and smooth, which is exactly what avoids the
"oscillation on later laps" problem.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from apartment_complex_map import build as build_apartment
from slam2d import FrontierExplorer, OccupancyGrid, FREE
from grid_planner import GridPlanner
from delivery_planner import DeliveryPlanner
from pedestrians2d import Pedestrians

_HERE = os.path.dirname(os.path.abspath(__file__))
MAP_NPZ = os.path.join(_HERE, "runs", "slam_apartment_complex.npz")

ROBOT_R = 0.22
DT = 0.1
MAX_SPEED = 0.9        # m/s
MAX_TURN = 2.2         # rad/s


# ===================== PHASE 1: blind SLAM + save ========================= #
def build_and_save_map(seed: int = 0, coverage_target: float = 0.97,
                       save_path: str = MAP_NPZ) -> Dict:
    """Robot with NO prior map frontier-explores apartment_complex_v1, then saves
    the discovered occupancy grid."""
    m = build_apartment()
    world = m["world"]
    ex = FrontierExplorer(world, cell=0.3, lidar_n=24, lidar_range=5.0,
                          step=0.12, max_steps=20000)
    # explore until frontiers are exhausted (don't stop early on a possibly-
    # mismatched coverage metric); coverage is reported afterwards vs the truth.
    scan = ex.explore(m["dock"], reachable_cells=None, record=True)
    occ = scan["grid"]
    # report true coverage: fraction of ground-truth reachable cells now FREE
    truth = _reachable_cells_on(occ, m)
    got = sum(1 for rc in truth if occ.state(*rc) == FREE)
    scan["coverage"] = got / max(len(truth), 1)
    free = occ.free_mask().astype(np.uint8)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, grid=free, cell=occ.cell, origin=np.array([occ.xmin, occ.ymin]),
             dock=np.array(m["dock"]),
             points=np.array([m["points"][k] for k in sorted(m["points"])]),
             point_ids=np.array(sorted(m["points"])))
    return {"map": m, "occ": occ, "coverage": scan["coverage"],
            "explore_trail": scan["trail"], "explore_steps": scan["steps"],
            "frames": scan["frames"], "save_path": save_path}


def _reachable_cells_on(occ, m):
    """Ground-truth reachable cells on THIS occ grid (matched cell size)."""
    from collections import deque
    sc = occ.to_cell(*m["dock"])
    def free(rc):
        x, y = occ.to_world(*rc)
        return occ.in_bounds(*rc) and not m["world"].segment_hits_circle((x, y), 0.18)
    if not free(sc):
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                if free((sc[0]+dr, sc[1]+dc)): sc = (sc[0]+dr, sc[1]+dc); break
    seen = {sc}; q = deque([sc]); cells = []
    while q:
        cur = q.popleft(); cells.append(cur)
        for nr, nc in ((cur[0]+1,cur[1]),(cur[0]-1,cur[1]),(cur[0],cur[1]+1),(cur[0],cur[1]-1)):
            if (nr, nc) not in seen and free((nr, nc)): seen.add((nr, nc)); q.append((nr, nc))
    return cells


def world_from_saved_map(sm) -> "object":
    """Rebuild a World whose walls are the boundaries of the SLAM-discovered free
    space, so the robot drives/senses on the map IT built in Phase 1 (not the
    original ground-truth map)."""
    from world2d import World
    g = sm["grid"]; cell = sm["cell"]; ox, oy = sm["origin"]
    R, C = g.shape
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
    w = World(half_width=cell*2.0, style="slam_built")
    w.segments = segs
    w.bounds = (ox - cell, ox + C*cell + cell, oy - cell, oy + R*cell + cell)
    return w


def _densify(path, max_gap=0.25):
    """Insert intermediate points so consecutive waypoints are <= max_gap apart
    (A* line-of-sight simplification leaves long straight gaps)."""
    if len(path) < 2:
        return [tuple(p) for p in path]
    out = [tuple(path[0])]
    for a, b in zip(path[:-1], path[1:]):
        a = np.array(a, float); b = np.array(b, float)
        d = float(np.linalg.norm(b - a)); n = max(1, int(math.ceil(d / max_gap)))
        for k in range(1, n + 1):
            out.append(tuple(a + (b - a) * k / n))
    return out


def load_saved_map(path: str = MAP_NPZ) -> Dict:
    d = np.load(path, allow_pickle=True)
    pts = {int(i): tuple(p) for i, p in zip(d["point_ids"], d["points"])}
    return {"grid": d["grid"], "cell": float(d["cell"]), "origin": tuple(d["origin"]),
            "dock": tuple(d["dock"]), "points": pts}


# ===================== smooth controller (pure-pursuit) =================== #
class SmoothDriver:
    """Pure-pursuit: steer toward a look-ahead point on the planned path with a
    rate-limited heading change -> smooth, non-weaving motion."""
    def __init__(self, lookahead=0.7, turn_rate_limit=0.35, arrive=0.45):
        self.lookahead = lookahead            # shorter -> hugs tight corridors / doors
        self.turn_limit = turn_rate_limit     # max |Δheading| per step (rad) -> smoothness
        self.arrive = arrive

    def lookahead_point(self, pos, path, seg_i):
        # advance the path index past points we've passed
        while seg_i < len(path) - 1 and \
                np.linalg.norm(np.array(path[seg_i]) - pos) < self.lookahead:
            seg_i += 1
        acc = 0.0; prev = np.array(pos, dtype=float)
        for j in range(seg_i, len(path)):
            q = np.array(path[j]); d = float(np.linalg.norm(q - prev))
            if acc + d >= self.lookahead:
                t = (self.lookahead - acc) / max(d, 1e-6)
                return prev + (q - prev) * t, seg_i
            acc += d; prev = q
        return np.array(path[-1]), seg_i


# ============ Phase 2 & 3 (RL): Hybrid A* + trained ms_mixed policy ======= #
def deliver_rl(picks: List[int], saved_map: Optional[Dict] = None, n_peds: int = 0,
               seed: int = 0, model_name: str = "ms_mixed", max_steps: int = 3000) -> Dict:
    """Hybrid controller using the TRAINED RL policy as the local driver:
    the global planner (TSP+A*) sets the route on the SLAM-built map, and the
    RecurrentPPO `ms_mixed` policy (sensor-only, the one we trained) drives it.
    Runs inside MultiStopEnv on a custom map = the map the robot built in Phase 1.
    """
    import os
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from multistop_env import MultiStopEnv

    sm = saved_map or load_saved_map()
    world = world_from_saved_map(sm)
    custom = {"world": world, "dock": tuple(sm["dock"]), "points": sm["points"],
              "grid": sm["grid"], "cell": sm["cell"], "origin": sm["origin"],
              "name": "slam_built"}

    model = RecurrentPPO.load(os.path.join(_HERE, "runs", f"{model_name}.zip"))
    vec = VecNormalize.load(os.path.join(_HERE, "runs", f"{model_name}_vecnorm.pkl"),
                            DummyVecEnv([lambda: MultiStopEnv(config=dict(procedural=True))]))
    mean = vec.obs_rms.mean.astype(np.float32); var = vec.obs_rms.var.astype(np.float32)
    norm = lambda o: np.clip((o - mean) / np.sqrt(var + vec.epsilon),
                             -vec.clip_obs, vec.clip_obs).astype(np.float32)

    env = MultiStopEnv(config=dict(n_lidar=24, lidar_range=5.0, lookahead=1.6,
                                   max_steps=max_steps, grace_steps=18, collision_grace=25,
                                   reverse_frac=0.4, procedural=False, domain_random=False,
                                   n_ped=n_peds, lidar_stack=1))
    o, info = env.reset(seed=seed, options={"custom_map": custom, "points": list(picks)})
    state = None; es = np.ones(1, bool); done = False
    trail = [tuple(env.pos)]; pedtrail = [[tuple(p) for p in env.peds.pos]] if env.peds else []
    stops = 0; dock_ok = False; coll = 0
    last_pos = np.array(env.pos); stall = 0; geo_steps = 0; geo_used = 0

    def geo_action():
        """Geometric pure-pursuit action toward the env's current look-ahead
        point on the A* route -> used to push through spots where RL stalls."""
        look = env._lookahead_point()
        d = look - env.pos
        bearing = (math.atan2(d[1], d[0]) - env.heading + math.pi) % (2*math.pi) - math.pi
        turn = float(np.clip(bearing / env.max_turn, -1, 1))
        fwd = 1.0 if abs(bearing) < 0.6 else 0.2     # slow into sharp turns
        return np.array([fwd, turn], dtype=np.float32)

    while not done:
        if geo_steps > 0:                            # currently in geo-assist burst
            a0 = geo_action(); geo_steps -= 1
        else:
            a, state = model.predict(norm(o)[None], state=state,
                                     episode_start=es, deterministic=True)
            a0 = a[0]
        es = np.zeros(1, bool)
        o, r, t, tr, inf = env.step(a0)
        # stall detection -> trigger a short geometric-assist burst
        moved = float(np.linalg.norm(np.array(env.pos) - last_pos)); last_pos = np.array(env.pos)
        if moved < 0.02:
            stall += 1
            if stall >= 8 and geo_steps == 0:
                geo_steps = 40; geo_used += 1; stall = 0
        else:
            stall = 0
        stops = max(stops, inf["stops_done"]); dock_ok = dock_ok or inf["arrived_dock"]
        coll += int(inf["collision"])
        trail.append(tuple(env.pos))
        if env.peds: pedtrail.append([tuple(p) for p in env.peds.pos])
        done = t or tr
    return {"reachable": True, "order": [int(x) for x in info["order"]],
            "delivered": stops, "stops_total": env.stops_total - 1,
            "returned_dock": dock_ok, "ped_hits": coll, "steps": env.step_i,
            "geo_assist_bursts": geo_used,
            "trail": trail, "pedtrail": pedtrail, "saved_map": sm, "world": world,
            "picks": list(picks), "controller": "Hybrid RL (ms_mixed) + geo-assist on SLAM map"}


# ===================== Phase 2 & 3 (geometric fallback) ================== #
def deliver(picks: List[int], saved_map: Optional[Dict] = None, world=None,
            n_peds: int = 0, seed: int = 0,
            avoid_radius: float = 1.0, avoid_slow: float = 0.45,
            avoid_strength: float = 0.6, max_steps: int = 20000) -> Dict:
    """Drive dock -> (TSP order of picks) -> dock on the saved map.

    People avoidance (n_peds>0): if a person's LiDAR-detected distance ahead is
    below `avoid_radius` (kept SMALL so the dodge is gentle, not a wide swerve),
    the robot slows to `avoid_slow` and adds a small lateral offset to the
    look-ahead target proportional to `avoid_strength`; once the person clears,
    the offset decays to zero and it eases straight back onto the path (no
    oscillation)."""
    sm = saved_map or load_saved_map()
    if world is None:
        # drive & sense on the map the robot BUILT in Phase 1 (SLAM grid), not the
        # original ground-truth map.
        world = world_from_saved_map(sm)
    dock = np.array(sm["dock"]); pts = sm["points"]

    # A* with an INFLATION layer -> waypoints down the corridor CENTRE (clearance
    # from walls); fall back to the raw grid for niches the inflation closes off.
    from hybrid_nav import inflate_map
    g_infl = inflate_map(sm["grid"], 1)
    planner = GridPlanner({"grid": g_infl, "cell": sm["cell"], "origin": sm["origin"]}, inflate=0)
    planner_raw = GridPlanner({"grid": sm["grid"], "cell": sm["cell"], "origin": sm["origin"]}, inflate=0)
    dp = DeliveryPlanner({"world": world, "dock": tuple(dock), "points": pts,
                          "grid": sm["grid"], "cell": sm["cell"], "origin": sm["origin"]})
    dp.planner = planner_raw
    plan = dp.optimize(list(picks))
    order = plan["order"]
    legs = []
    prev = tuple(dock)
    for s in list(order) + ["dock"]:
        tgt = tuple(dock) if s == "dock" else pts[s]
        wps, _ = planner.plan(prev, tgt)
        if wps is None:
            wps, _ = planner_raw.plan(prev, tgt)
        if wps is None:
            return {"reachable": False, "order": order}
        legs.append(wps); prev = tgt
    route_pts = [p for leg in legs for p in leg]   # full A* route for plotting

    peds = Pedestrians(world, np.random.default_rng(seed), n=n_peds,
                       speed_range=(0.5, 1.0), dt=DT,
                       grid_map={"grid": sm["grid"], "cell": sm["cell"], "origin": sm["origin"]}) \
        if n_peds > 0 else None

    def path_cell(path, seg_i):
        return path[min(seg_i + 1, len(path) - 1)]

    drv = SmoothDriver()
    pos = dock.astype(float).copy(); heading = 0.0; stuck = 0
    trail = [tuple(pos)]; pedtrail = [[tuple(p) for p in peds.pos]] if peds else []
    lidar_angles = np.linspace(0, 2*math.pi, 24, endpoint=False)
    lat_offset = 0.0                  # current smoothed lateral dodge (m)
    delivered = []; steps = 0; ped_hits = 0; avoid_events = 0; avoid_pts = []

    targets = [pts[s] for s in order] + [tuple(dock)]
    labels = [str(s) for s in order] + ["dock"]
    actual_route = []                                   # what was really driven (for plotting)
    for tgt0, lab in zip(targets, labels):
        # RE-PLAN this leg from the robot's ACTUAL current position (so a slow /
        # imperfect previous leg never corrupts the next leg's path).
        wps, _ = planner.plan(tuple(pos), tuple(tgt0))
        if wps is None:
            wps, _ = planner_raw.plan(tuple(pos), tuple(tgt0))
        if wps is None:
            continue
        leg_path = wps
        actual_route.extend(leg_path)
        # densify the A* waypoints so we always follow the corridor centre line
        path = _densify(leg_path, max_gap=0.25)
        tgt = np.array(tgt0, float)
        wi = 1                                          # current waypoint index
        guard = 0; stall = 0; last = pos.copy()
        # drive until we ACTUALLY arrive at the leg target (not until waypoints
        # run out) so a stuck robot is never falsely counted as delivered.
        while np.linalg.norm(tgt - pos) > 0.35 and steps < max_steps and guard < 5000:
            guard += 1; steps += 1
            if peds: peds.step(DT)
            # advance the look-ahead index past waypoints we've reached
            while wi < len(path) - 1 and np.linalg.norm(np.array(path[wi]) - pos) < 0.4:
                wi += 1
            wp = np.array(path[wi])

            fwd = np.array([math.cos(heading), math.sin(heading)])
            left = np.array([-fwd[1], fwd[0]])

            # ---- people shield: SMALL lateral nudge + slow; STOP if blocked ----
            target_off = 0.0; slow = 1.0; blocked_by_person = False
            if peds is not None:
                best_d = 1e9; best_side = 0.0
                for pp in peds.pos:
                    rel = pp - pos; dahead = float(rel @ fwd)
                    side = float(rel @ left); d = float(np.linalg.norm(rel))
                    if dahead > -0.1 and d < avoid_radius and d < best_d:
                        best_d = d; best_side = side
                if best_d < avoid_radius:
                    avoid_events += 1; slow = avoid_slow; avoid_pts.append(tuple(pos))
                    target_off = -np.sign(best_side or 1.0) * (ROBOT_R + 0.25)
                    if best_d < ROBOT_R + 0.30:
                        blocked_by_person = True
            lat_offset += 0.18 * (target_off - lat_offset)

            aim = wp + left * lat_offset
            desired = math.atan2(aim[1] - pos[1], aim[0] - pos[0])
            dh = (desired - heading + math.pi) % (2*math.pi) - math.pi
            dh = max(-drv.turn_limit, min(drv.turn_limit, dh))
            heading = (heading + dh + math.pi) % (2*math.pi) - math.pi
            if blocked_by_person:
                trail.append(tuple(pos))
                if peds: pedtrail.append([tuple(p) for p in peds.pos])
                last = pos.copy(); continue
            speed = MAX_SPEED * slow * (1.0 - 0.4 * min(abs(dh)/drv.turn_limit, 1.0))
            npos = pos + np.array([math.cos(heading), math.sin(heading)]) * speed * DT
            if not world.segment_hits_circle(tuple(npos), ROBOT_R) and \
               not (peds is not None and peds.hits_robot(tuple(npos), ROBOT_R)):
                pos = npos
            # stuck detection -> recovery (fan headings toward next waypoint + reverse)
            if float(np.linalg.norm(pos - last)) < 0.01:
                stall += 1
            else:
                stall = 0
            last = pos.copy()
            if stall >= 15:
                nxt = np.array(path[min(wi + 1, len(path) - 1)])
                base = math.atan2(nxt[1] - pos[1], nxt[0] - pos[0])
                for off in (0.0, 0.5, -0.5, 1.0, -1.0, 1.6, -1.6, math.pi):
                    cand = pos + np.array([math.cos(base+off), math.sin(base+off)]) * (MAX_SPEED*0.5*DT)
                    if not world.segment_hits_circle(tuple(cand), ROBOT_R) and \
                       not (peds is not None and peds.hits_robot(tuple(cand), ROBOT_R)):
                        pos = cand; heading = (base+off+math.pi)%(2*math.pi)-math.pi
                        stall = 0; wi = min(wi+1, len(path)-1); break
            trail.append(tuple(pos))
            if peds: pedtrail.append([tuple(p) for p in peds.pos])
        # count as delivered ONLY if we truly reached the target
        if lab != "dock" and np.linalg.norm(tgt - pos) <= 0.5:
            delivered.append(int(lab))

    back = float(np.linalg.norm(pos - dock))
    return {"reachable": True, "order": [int(x) for x in order], "delivered": delivered,
            "returned_dock": back <= 0.6, "steps": steps, "ped_hits": ped_hits,
            "avoid_events": avoid_events, "trail": trail, "pedtrail": pedtrail,
            "avoid_pts": avoid_pts, "route": actual_route or route_pts,
            "saved_map": sm, "world": world, "picks": list(picks)}
