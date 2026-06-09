"""Delivery task: parcel->locker manifest, reward, termination and curriculum.

Curriculum levels (set in YAML under env.curriculum.level):
  L0  1 parcel, nearest locker to dock
  L1  1 parcel, random locker
  L2  multiple parcels, agent self-routes (current target = nearest remaining)
  L3  + static obstacles
  L4  + moving residents (pedestrian stub)
  L5  + domain randomization (friction/mass/sensor-noise/spawn)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ManifestEntry:
    parcel: int
    locker_id: int
    delivered: bool = False


class DeliveryTask:
    def __init__(self, config: dict, np_random: np.random.Generator):
        self.np_random = np_random
        self.rcfg = config["env"]["reward"]
        self.tcfg = config["env"]["task"]
        self.term = config["env"]["termination"]
        self.dock_zone_radius = float(config["env"]["mechanism"]["dock_zone_radius"])
        self.level = int(config["env"]["curriculum"]["level"])
        self.num_parcels_cfg = int(self.tcfg["num_parcels"])
        # optional per-run overrides (e.g. a pedestrian-only demo scenario)
        self.override = config["env"].get("scenario_override") or {}

        self.scene = None
        self.dock_xy = np.zeros(2, dtype=np.float32)
        self.manifest: List[ManifestEntry] = []
        self.num_lockers = 0
        self._locker_by_id: Dict[int, object] = {}

    # ------------------------------------------------------------------ #
    def reset(self, scene) -> None:
        self.scene = scene
        self.dock_xy = np.array(scene.dock_pos[:2], dtype=np.float32)
        self.num_lockers = len(scene.lockers)
        self._locker_by_id = {l.id: l for l in scene.lockers}

        # Optional pool restriction: only target lockers whose `side` is allowed
        # (e.g. ["north"] = main-corridor lockers only). Lets us train/compare on
        # the reliably reachable corridor destinations without the hard
        # room/arc lockers. Defaults to all sides.
        allowed_sides = self.override.get("locker_sides")
        if allowed_sides:
            pool = [l for l in scene.lockers if l.side in allowed_sides]
        else:
            pool = list(scene.lockers)
        if not pool:
            pool = list(scene.lockers)
        pool_ids = [l.id for l in pool]

        forced = self.override.get("force_locker_id")
        if forced is not None:
            # explicit destination (used by the 2D map review to send all three
            # models to the SAME locker for a fair comparison)
            chosen = [int(forced)]
        elif self.level <= 0:
            nearest = min(pool, key=lambda l: np.linalg.norm(np.array(l.dock) - self.dock_xy))
            chosen = [nearest.id]
        elif self.level == 1:
            chosen = [int(self.np_random.choice(pool_ids))]
        else:
            n = max(1, self.num_parcels_cfg)
            chosen = list(self.np_random.choice(pool_ids, size=min(n, len(pool_ids)),
                                                replace=False))

        self.manifest = [ManifestEntry(parcel=i, locker_id=int(lid))
                         for i, lid in enumerate(chosen)]

    # ------------------------------------------------------------------ #
    def episode_settings(self) -> Dict:
        level = self.level
        settings = {
            "num_obstacles": {3: 6, 4: 6, 5: 8}.get(level, 0),
            "num_pedestrians": {4: 3, 5: 4}.get(level, 0),
            "domain_random": level >= 5,
            "noise_scale": 1.5 if level >= 5 else 1.0,
        }
        settings.update({k: v for k, v in self.override.items() if k in settings})
        return settings

    # ------------------------------------------------------------------ #
    @property
    def parcels_carried(self) -> int:
        return sum(1 for m in self.manifest if not m.delivered)

    @property
    def all_delivered(self) -> bool:
        return all(m.delivered for m in self.manifest)

    def remaining_vector(self) -> np.ndarray:
        vec = np.zeros(self.num_lockers, dtype=np.float32)
        for m in self.manifest:
            if not m.delivered:
                vec[m.locker_id] = 1.0
        return vec

    def remaining_target_ids(self) -> List[int]:
        return [m.locker_id for m in self.manifest if not m.delivered]

    def current_target_xy(self, robot_xy: np.ndarray) -> Tuple[str, np.ndarray]:
        """Nearest not-yet-delivered locker dock, or the charging dock if done."""
        remaining = self.remaining_target_ids()
        if not remaining:
            return "dock", self.dock_xy.copy()
        best = min(remaining, key=lambda lid:
                   np.linalg.norm(np.array(self._locker_by_id[lid].dock) - robot_xy))
        return "locker", np.array(self._locker_by_id[best].dock, dtype=np.float32)

    def current_target_locker(self, robot_xy: np.ndarray):
        """The locker object the robot is currently routing to, or None if the
        remaining target is the charging dock (used to draw the red marker)."""
        remaining = self.remaining_target_ids()
        if not remaining:
            return None
        best = min(remaining, key=lambda lid:
                   np.linalg.norm(np.array(self._locker_by_id[lid].dock) - robot_xy))
        return self._locker_by_id[best]

    def docked_locker_id(self, robot_xy: np.ndarray) -> Optional[int]:
        # Only a locker that still owes an undelivered parcel can be docked at,
        # and we pick the NEAREST such locker within the dock zone. This avoids
        # docking at the wrong locker (e.g. the opposite-side one whose dock
        # point can fall inside the same zone) and guarantees deliver() succeeds.
        remaining = set(self.remaining_target_ids())
        best, best_d = None, self.dock_zone_radius
        for locker in self.scene.lockers:
            if locker.id not in remaining:
                continue
            d = float(np.linalg.norm(np.array(locker.dock) - robot_xy))
            if d <= best_d:
                best, best_d = locker.id, d
        return best

    def locker_by_id(self, lid: int):
        return self._locker_by_id[lid]

    def deliver(self, locker_id: int) -> bool:
        """Returns True if this was a CORRECT (remaining-manifest) delivery."""
        for m in self.manifest:
            if m.locker_id == locker_id and not m.delivered:
                m.delivered = True
                return True
        return False

    def at_dock(self, robot_xy: np.ndarray) -> bool:
        return float(np.linalg.norm(robot_xy - self.dock_xy)) <= self.dock_zone_radius

    # ------------------------------------------------------------------ #
    def compute_reward(self, events: Dict) -> float:
        r = self.rcfg["time_penalty"]
        r += self.rcfg["progress"] * events.get("progress", 0.0)
        r += self.rcfg["energy_penalty"] * events.get("distance_moved", 0.0)
        r += self.rcfg["vibration_penalty"] * events.get("d_omega", 0.0)
        if events.get("collision_world"):
            r += self.rcfg["collision_penalty"]
        if events.get("collision_pedestrian"):
            r += self.rcfg["pedestrian_collision_penalty"]
        if events.get("delivered"):
            r += self.rcfg["delivery_success"]
        if events.get("wrong_delivery"):
            r += self.rcfg["wrong_delivery_penalty"]
        if events.get("all_done"):
            r += self.rcfg["all_done_bonus"]
        if events.get("returned_dock"):
            r += self.rcfg["return_dock_bonus"]
        if events.get("tilt"):
            r += self.rcfg["tilt_penalty"]
        return float(r)

    def check_termination(self, events: Dict) -> Tuple[bool, str]:
        if events.get("returned_dock") and self.all_delivered:
            return True, "success"
        if events.get("tilt"):
            return True, "tilt"
        if self.term.get("on_collision", True) and (
            events.get("collision_world") or events.get("collision_pedestrian")
        ):
            return True, "collision"
        return False, ""
