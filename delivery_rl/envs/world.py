"""Builds the apartment-corridor world in PyBullet.

Floorplan (procedural fallback, ``map_style`` in config):
  * "rich" (default): a WIDE main corridor along x, several deep side ROOMS
    ("ngõ ngách"), and a curved ARC branch ("hành lang vòng cung") on the south
    side. Lockers sit flush to walls in the corridor, inside the rooms, and along
    the arc -- so destinations are spread across genuinely different sub-areas.
  * "straight": the original narrow single corridor (kept for comparison).

All static blockers are recorded as 2D AABBs in ``blocked_rects`` (and richer
``draw_rects`` for plotting) so the path planner and the 2D top-down review can
work directly from geometry -- which makes different maps look clearly different.

TODO(asset): drop a real ``assets/scene_map.json`` (same schema as the emitted
``assets/scene_map.generated.json``) and/or URDF/USD meshes here.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pybullet as p


@dataclass
class Locker:
    id: int
    pos: Tuple[float, float, float]      # centre of the locker box
    yaw: float                           # facing into the corridor
    dock: Tuple[float, float]            # (x, y) approach point in the corridor
    side: str                            # "north" / "south" / "room" / "arc"
    body_id: int = -1


@dataclass
class SceneMap:
    corridor_length: float
    corridor_width: float
    dock_pos: Tuple[float, float, float]
    dock_yaw: float
    lockers: List[Locker] = field(default_factory=list)
    bounds: Tuple[float, float, float, float] = (-20, 20, -8, 6)  # xmin,xmax,ymin,ymax

    def to_json(self) -> dict:
        return {
            "corridor": {"length": self.corridor_length, "width": self.corridor_width},
            "bounds": list(self.bounds),
            "dock": {"pos": list(self.dock_pos), "yaw": self.dock_yaw},
            "lockers": [
                {"id": l.id, "pos": list(l.pos), "yaw": l.yaw,
                 "dock": list(l.dock), "side": l.side}
                for l in self.lockers
            ],
        }

    @staticmethod
    def from_json(data: dict) -> "SceneMap":
        sm = SceneMap(
            corridor_length=data["corridor"]["length"],
            corridor_width=data["corridor"]["width"],
            dock_pos=tuple(data["dock"]["pos"]),
            dock_yaw=data["dock"].get("yaw", 0.0),
            bounds=tuple(data.get("bounds", (-20, 20, -8, 6))),
        )
        for l in data["lockers"]:
            sm.lockers.append(Locker(id=l["id"], pos=tuple(l["pos"]), yaw=l["yaw"],
                                     dock=tuple(l["dock"]), side=l.get("side", "north")))
        return sm


class CorridorWorld:
    def __init__(self, client: int, config: dict, np_random: np.random.Generator,
                 base_dir: str):
        self.client = client
        self.cfg = config["env"]["world"]
        self.np_random = np_random
        self.base_dir = base_dir
        self.map_style = self.cfg.get("map_style", "rich")
        self.wall_h = float(self.cfg["wall_height"])
        self.wall_t = float(self.cfg["wall_thickness"])

        self.floor_id: int = -1
        self.collision_body_ids: List[int] = []
        self.obstacle_body_ids: List[int] = []
        self.obstacles_geom: List[Tuple[float, float, float]] = []
        self.locker_body_to_id: Dict[int, int] = {}
        # 2D AABBs (cx, cy, hx, hy) of every static blocker -- used by the planner
        self.blocked_rects: List[Tuple[float, float, float, float]] = []
        # richer drawing data: (cx, cy, hx, hy, kind)  kind in wall/room/elevator/stair
        self.draw_rects: List[Tuple[float, float, float, float, str]] = []
        self.scene: Optional[SceneMap] = None
        self._built = False

    # ------------------------------------------------------------------ #
    def build(self) -> SceneMap:
        if self._built:
            return self.scene
        scene_path = os.path.join(self.base_dir, self.cfg["scene_map"])
        if os.path.isfile(scene_path):
            with open(scene_path, "r", encoding="utf-8") as f:
                self.scene = SceneMap.from_json(json.load(f))
            self._build_floor(self.scene.bounds)
            # NOTE: loading a custom scene_map provides lockers/bounds; walls for a
            # custom map should be added via meshes (TODO asset). We still place
            # locker bodies so collisions/markers work.
            self._build_lockers(self.scene)
        else:
            if self.map_style == "straight":
                self.scene = self._procedural_straight()
            else:
                self.scene = self._procedural_rich()
            self._export_generated(self.scene)
        self._built = True
        return self.scene

    # ------------------------------------------------------------------ #
    #  primitive helpers (also record AABB for the planner / 2D plot)
    # ------------------------------------------------------------------ #
    def _add_box(self, half_extents, pos, rgba, mass=0.0, collision=True) -> int:
        col = (p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents,
                                      physicsClientId=self.client) if collision else -1)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba,
                                  physicsClientId=self.client)
        return p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col,
                                 baseVisualShapeIndex=vis, basePosition=pos,
                                 physicsClientId=self.client)

    def _wall(self, cx, cy, hx, hy, kind="wall", rgba=(0.55, 0.57, 0.62, 1.0)) -> int:
        body = self._add_box([hx, hy, self.wall_h / 2], [cx, cy, self.wall_h / 2], list(rgba))
        self.collision_body_ids.append(body)
        self.blocked_rects.append((cx, cy, hx, hy))
        self.draw_rects.append((cx, cy, hx, hy, kind))
        return body

    def _build_floor(self, bounds) -> None:
        xmin, xmax, ymin, ymax = bounds
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        hx, hy = (xmax - xmin) / 2 + 1.0, (ymax - ymin) / 2 + 1.0
        self.floor_id = self._add_box([hx, hy, 0.02], [cx, cy, -0.02],
                                      [0.18, 0.19, 0.22, 1.0])

    # ------------------------------------------------------------------ #
    #  RICH floorplan: wide corridor + rooms + arc branch
    # ------------------------------------------------------------------ #
    def _procedural_rich(self) -> SceneMap:
        t = self.wall_t
        hw = float(self.cfg.get("corridor_width", 3.0)) / 2.0   # main corridor half-width
        x0 = -float(self.cfg.get("corridor_length", 36.0)) / 2.0
        x1 = -x0
        room_depth = float(self.cfg.get("room_depth", 2.8))
        room_w = float(self.cfg.get("room_width", 2.2))
        lw, ldp, lh = self.cfg["locker_size"]
        base_h = self.cfg["locker_base_height"]
        lz = base_h + lh / 2.0

        # --- north rooms (x centres) and south rooms ---
        north_rooms = [-12.0, -5.0, 5.0, 12.0]
        south_rooms = [9.0]            # south side mostly used by the arc
        # --- arc on the south side: ring between r_in and r_out, lower half ---
        arc_cx, arc_cy = -10.0, -hw
        r_in = float(self.cfg.get("arc_inner_radius", 3.0))
        r_out = r_in + float(self.cfg.get("arc_width", 2.2))
        arc_left, arc_right = arc_cx - r_out, arc_cx + r_out   # mouths on south wall

        bounds = (x0 - 1.5, x1 + 1.5, -(r_out + abs(arc_cy) + 1.5), hw + room_depth + 1.5)

        self._build_floor(bounds)

        # ---- main corridor north & south walls with gaps ----
        north_gaps = [(cx, room_w) for cx in north_rooms]
        # elevator gap at centre-north
        south_gaps = [(cx, room_w) for cx in south_rooms]
        south_gaps += [(arc_left + (r_out - r_in) / 2, r_out - r_in),   # left arc mouth
                       (arc_right - (r_out - r_in) / 2, r_out - r_in)]  # right arc mouth
        self._wall_strip_x(+hw, x0, x1, north_gaps, "wall")
        self._wall_strip_x(-hw, x0, x1, south_gaps, "wall")
        # end caps
        self._wall((x0 + x1) / 2 * 0 + x0, 0, t / 2, hw, "wall")
        self._wall(x1, 0, t / 2, hw, "wall")

        lockers: List[Locker] = []
        lid = 0

        # ---- lockers flush on main corridor walls (between the room gaps) ----
        corridor_locker_x = [-16.0, -8.5, -1.5, 1.5, 8.5, 16.0]
        for cx in corridor_locker_x:
            # north wall locker (faces south, into corridor)
            lockers.append(Locker(lid, (cx, hw - ldp / 2 - 0.02, lz), -math.pi / 2,
                                  (cx, hw - ldp - 0.5), "north")); lid += 1
        # ---- rooms: build walls + a locker at the back of each room ----
        for cx in north_rooms:
            self._room(cx, +1, hw, room_w, room_depth)
            ly = hw + room_depth - ldp / 2 - 0.1
            lockers.append(Locker(lid, (cx, ly, lz), -math.pi / 2,
                                  (cx, hw + room_depth - ldp - 0.7), "room")); lid += 1
        for cx in south_rooms:
            self._room(cx, -1, hw, room_w, room_depth)
            ly = -(hw + room_depth - ldp / 2 - 0.1)
            lockers.append(Locker(lid, (cx, ly, lz), math.pi / 2,
                                  (cx, -(hw + room_depth - ldp - 0.7)), "room")); lid += 1

        # ---- arc branch: ring walls + lockers along the outer wall ----
        self._arc_ring(arc_cx, arc_cy, r_in, r_out)
        for ang_deg in (215, 245, 270, 295, 325):
            a = math.radians(ang_deg)
            # locker sits just inside the outer wall, facing the ring centre
            lr = r_out - ldp / 2 - 0.08
            lx = arc_cx + lr * math.cos(a)
            ly = arc_cy + lr * math.sin(a)
            dock_r = r_in + (r_out - r_in) * 0.5
            dx = arc_cx + dock_r * math.cos(a)
            dy = arc_cy + dock_r * math.sin(a)
            yaw = math.atan2(arc_cy - ly, arc_cx - lx)
            lockers.append(Locker(lid, (lx, ly, lz), yaw, (dx, dy), "arc")); lid += 1

        # elevator block at centre-north + charging dock at centre
        self._wall(0.0, hw + 0.25, 1.0, 0.25, "elevator", rgba=(0.30, 0.45, 0.60, 1.0))
        dock_pos = (0.0, 0.0, 0.0)
        self._add_box([0.35, 0.35, 0.02], [0, 0, 0.02], [0.2, 0.8, 0.4, 1.0], collision=False)

        scene = SceneMap(corridor_length=(x1 - x0), corridor_width=2 * hw,
                         dock_pos=dock_pos, dock_yaw=0.0, lockers=lockers, bounds=bounds)
        self._build_lockers(scene)
        return scene

    def _wall_strip_x(self, y, x0, x1, gaps, kind) -> None:
        """Wall along x at height y, with gaps (centre, width) removed."""
        t = self.wall_t
        intervals = [(x0, x1)]
        for gc, gw in sorted(gaps):
            g0, g1 = gc - gw / 2, gc + gw / 2
            new = []
            for a, b in intervals:
                if g1 <= a or g0 >= b:
                    new.append((a, b))
                else:
                    if a < g0:
                        new.append((a, g0))
                    if g1 < b:
                        new.append((g1, b))
            intervals = new
        for a, b in intervals:
            if b - a < 1e-3:
                continue
            self._wall((a + b) / 2, y, (b - a) / 2, t / 2, kind)

    def _room(self, cx, side, hw, room_w, depth) -> None:
        """A rectangular room ("ngõ ngách") opening off the corridor at x=cx."""
        t = self.wall_t
        y_in = side * hw
        y_out = side * (hw + depth)
        # two side walls
        for sx in (cx - room_w / 2, cx + room_w / 2):
            self._wall(sx, side * (hw + depth / 2), t / 2, depth / 2, "room")
        # back wall
        self._wall(cx, y_out, room_w / 2 + t / 2, t / 2, "room")

    def _arc_ring(self, cx, cy, r_in, r_out, n_seg=22) -> None:
        """Lower-half ring (semicircle) made of short wall segments -> a curved
        drivable corridor between r_in and r_out."""
        t = self.wall_t
        for r, kind in ((r_in, "wall"), (r_out, "wall")):
            for k in range(n_seg):
                a0 = math.pi + math.pi * k / n_seg
                a1 = math.pi + math.pi * (k + 1) / n_seg
                am = (a0 + a1) / 2
                seg_len = r * (math.pi / n_seg)
                sx = cx + r * math.cos(am)
                sy = cy + r * math.sin(am)
                # approximate the curved wall with a small box tangent to the arc
                self._add_box([seg_len / 2 + t, t / 2, self.wall_h / 2],
                              [sx, sy, self.wall_h / 2], [0.55, 0.57, 0.62, 1.0])
                # record an AABB roughly covering the segment for the planner
                self.blocked_rects.append((sx, sy, max(seg_len / 2, t), t))
                self.draw_rects.append((sx, sy, seg_len / 2, t / 2, "wall"))
                self.collision_body_ids.append(self.collision_body_ids and -1 or -1)
        # (collision bodies already created via _add_box above; ids tracked loosely)

    # ------------------------------------------------------------------ #
    #  STRAIGHT floorplan (original, kept for comparison)
    # ------------------------------------------------------------------ #
    def _procedural_straight(self) -> SceneMap:
        t = self.wall_t
        hw = 1.0
        x0, x1 = -22.0, 22.0
        bounds = (x0 - 1.5, x1 + 1.5, -3.0, 3.0)
        self._build_floor(bounds)
        self._wall_strip_x(+hw, x0, x1, [], "wall")
        self._wall_strip_x(-hw, x0, x1, [], "wall")
        self._wall(x0, 0, t / 2, hw, "wall")
        self._wall(x1, 0, t / 2, hw, "wall")
        lw, ldp, lh = self.cfg["locker_size"]
        lz = self.cfg["locker_base_height"] + lh / 2.0
        lockers = []
        xs = np.linspace(x0 + 2, x1 - 2, 10)
        lid = 0
        for cx in xs:
            lockers.append(Locker(lid, (float(cx), hw - ldp / 2 - 0.02, lz), -math.pi / 2,
                                  (float(cx), hw - ldp - 0.4), "north")); lid += 1
        scene = SceneMap((x1 - x0), 2 * hw, (0.0, 0.0, 0.0), 0.0, lockers, bounds)
        self._build_lockers(scene)
        return scene

    # ------------------------------------------------------------------ #
    def _build_lockers(self, scene: SceneMap) -> None:
        lw, ldp, lh = self.cfg["locker_size"]
        for locker in scene.lockers:
            # orient the locker box to face the corridor (yaw)
            orn = p.getQuaternionFromEuler([0, 0, locker.yaw + math.pi / 2])
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[lw / 2, ldp / 2, lh / 2],
                                         physicsClientId=self.client)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[lw / 2, ldp / 2, lh / 2],
                                      rgbaColor=[0.75, 0.6, 0.35, 1.0],
                                      physicsClientId=self.client)
            body = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col,
                                     baseVisualShapeIndex=vis, basePosition=list(locker.pos),
                                     baseOrientation=orn, physicsClientId=self.client)
            locker.body_id = body
            self.collision_body_ids.append(body)
            self.locker_body_to_id[body] = locker.id
            # AABB (axis-aligned envelope) for the planner + drawing
            hx = max(lw, ldp) / 2
            self.blocked_rects.append((locker.pos[0], locker.pos[1], hx, hx))

    def _export_generated(self, scene: SceneMap) -> None:
        out = os.path.join(self.base_dir, "assets", "scene_map.generated.json")
        try:
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(scene.to_json(), f, indent=2)
        except OSError:
            pass

    # ------------------------------------------------------------------ #
    #  static obstacles (curriculum L3) + free-space test for the planner
    # ------------------------------------------------------------------ #
    def reset_obstacles(self, num_obstacles: int) -> None:
        for body in self.obstacle_body_ids:
            p.removeBody(body, physicsClientId=self.client)
        self.obstacle_body_ids = []
        self.obstacles_geom = []
        if num_obstacles <= 0:
            return
        xmin, xmax, ymin, ymax = self.scene.bounds
        hw = self.scene.corridor_width / 2.0
        for _ in range(num_obstacles):
            # drop obstacles inside the main corridor lane
            x = float(self.np_random.uniform(xmin + 4, xmax - 4))
            y = float(self.np_random.uniform(-hw + 0.4, hw - 0.4))
            s = float(self.np_random.uniform(0.12, 0.25))
            body = self._add_box([s, s, 0.3], [x, y, 0.3], [0.85, 0.4, 0.3, 1.0])
            self.obstacle_body_ids.append(body)
            self.obstacles_geom.append((x, y, s))

    def is_free(self, x: float, y: float, clearance: float = 0.0) -> bool:
        """Geometric free-space test (point vs all static AABBs, inflated by
        ``clearance``). Used by the path planner only; sim collisions stay
        PyBullet-based."""
        xmin, xmax, ymin, ymax = self.scene.bounds
        if x < xmin or x > xmax or y < ymin or y > ymax:
            return False
        for (cx, cy, hx, hy) in self.blocked_rects:
            if abs(x - cx) <= hx + clearance and abs(y - cy) <= hy + clearance:
                return False
        for (ox, oy, s) in self.obstacles_geom:
            if abs(x - ox) <= s + clearance and abs(y - oy) <= s + clearance:
                return False
        return True

    @property
    def all_collision_ids(self) -> List[int]:
        return [b for b in (self.collision_body_ids + self.obstacle_body_ids) if b >= 0]
