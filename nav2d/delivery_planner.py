"""Multi-stop delivery route optimizer (TSP) for the fixed maps.

Given the dock and a set of chosen delivery point ids (1-3, but works for more),
compute the visiting ORDER that minimises total **real travel distance**
(dock -> p_a -> p_b -> p_c -> dock), using the grid planner's true path lengths
("tiện đường", not straight-line). For <=8 stops we enumerate all permutations
(optimal); above that we fall back to nearest-neighbour + 2-opt.

The robot delivers ALL parcels before returning to the dock (battery-friendly:
one trip, shortest closed tour). We also report an estimated battery use from the
total distance.

Returns a plan: ordered point ids, the full waypoint path (concatenated legs),
per-leg distances, total distance and battery estimate.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Tuple

from grid_planner import GridPlanner


class DeliveryPlanner:
    def __init__(self, map_dict: Dict, battery_per_m: float = 0.4):
        self.map = map_dict
        self.dock = map_dict["dock"]
        self.points = map_dict["points"]
        self.planner = GridPlanner(map_dict)
        self.battery_per_m = battery_per_m   # % battery per metre (tune for demo)
        self._dist_cache: Dict[Tuple, float] = {}

    def _dist(self, a_xy, b_xy) -> float:
        key = (round(a_xy[0], 3), round(a_xy[1], 3), round(b_xy[0], 3), round(b_xy[1], 3))
        if key not in self._dist_cache:
            self._dist_cache[key] = self.planner.distance(a_xy, b_xy)
        return self._dist_cache[key]

    def _tour_len(self, order: List[int]) -> float:
        """dock -> order... -> dock total length."""
        total = 0.0
        prev = self.dock
        for pid in order:
            total += self._dist(prev, self.points[pid]); prev = self.points[pid]
        total += self._dist(prev, self.dock)   # return home
        return total

    def optimize(self, chosen_ids: List[int]) -> Dict:
        ids = list(dict.fromkeys(chosen_ids))   # unique, keep order
        for pid in ids:
            if pid not in self.points:
                raise ValueError(f"unknown delivery point id {pid}")
        if not ids:
            return {"order": [], "total_dist": 0.0, "legs": [], "battery_pct": 0.0,
                    "waypoints": [self.dock], "reachable": True}

        if len(ids) <= 8:
            best_order, best_len = None, float("inf")
            for perm in itertools.permutations(ids):
                L = self._tour_len(list(perm))
                if L < best_len:
                    best_len, best_order = L, list(perm)
        else:
            best_order = self._nn_2opt(ids); best_len = self._tour_len(best_order)

        # no finite tour (some leg unreachable) -> report unreachable cleanly
        if best_order is None or not math.isfinite(best_len):
            return {"order": list(ids), "reachable": False, "total_dist": float("inf"),
                    "legs": [], "battery_pct": float("inf"), "waypoints": [self.dock]}

        # build the full waypoint path (dock -> ... -> dock) + per-leg info
        waypoints: List[Tuple[float, float]] = [self.dock]
        legs = []
        stops = best_order + ["dock"]
        prev = self.dock; prev_name = "dock"
        for s in stops:
            tgt = self.dock if s == "dock" else self.points[s]
            wps, L = self.planner.plan(prev, tgt)
            if wps is None:
                return {"order": best_order, "reachable": False, "total_dist": float("inf"),
                        "legs": legs, "battery_pct": float("inf"), "waypoints": waypoints}
            waypoints.extend(wps[1:])   # skip duplicate start
            legs.append({"from": prev_name, "to": s, "dist": L})
            prev, prev_name = tgt, str(s)

        return {"order": best_order, "total_dist": best_len, "legs": legs,
                "battery_pct": best_len * self.battery_per_m, "waypoints": waypoints,
                "reachable": True}

    def _nn_2opt(self, ids):
        # nearest-neighbour seed
        unv = set(ids); order = []; cur = self.dock
        while unv:
            nxt = min(unv, key=lambda p: self._dist(cur, self.points[p]))
            order.append(nxt); unv.discard(nxt); cur = self.points[nxt]
        # 2-opt
        improved = True
        while improved:
            improved = False
            for i in range(len(order) - 1):
                for j in range(i + 1, len(order)):
                    cand = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                    if self._tour_len(cand) + 1e-9 < self._tour_len(order):
                        order = cand; improved = True
        return order
