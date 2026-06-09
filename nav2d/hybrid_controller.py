"""BƯỚC 3 + 5 — Hybrid Controller cốt lõi.

Thành phần:
  * AStarPlanner  : có inflate_map (bơm phồng vật cản ~0.3m). plan() trả None nếu
                    không có đường (ví dụ ngách quá hẹp sau khi inflate).
  * MissionController : quản lý danh sách trạm. Với mỗi trạm: gọi A* từ vị trí
                    HIỆN TẠI -> nếu None thì HỦY trạm (đếm vào cancelled) và tự
                    tính đường tới trạm kế tiếp. Sau khi giao hết -> về Dock.
  * Stuck Detector + Dynamic Replanning ("Google Maps reroute"): nếu robot ít
                    dịch chuyển trong cửa sổ steps -> đánh dấu ô đang đứng là vật
                    cản TẠM trên một bản sao lưới -> replan A* vẽ đường vòng.

Lái cục bộ: pure-pursuit bám điểm look-ahead trên tuyến A*; SafetyShield override
khi có người. (RL ms_mixed cũng có thể cắm vào qua hook `local_policy`, mặc định
pure-pursuit cho ổn định.)
"""

from __future__ import annotations

import heapq
import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from world2d import World
from safety_shield import SafetyShield

DT = 0.1
ROBOT_R = 0.22
MAX_SPEED = 0.9
MAX_TURN = 2.2


# ============================ A* planner ================================== #
def inflate_map(grid: np.ndarray, layers: int = 1) -> np.ndarray:
    """Bơm phồng vật cản: co vùng free `layers` ô quanh mọi tường (~0.3m/ô)."""
    g = grid.copy(); R, C = g.shape
    for _ in range(layers):
        free = g == 1; keep = free.copy()
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


class AStarPlanner:
    def __init__(self, grid, cell, origin, inflate_layers=1):
        self.cell = cell; self.origin = origin
        self.base = grid.astype(np.uint8)
        self.inflate_layers = inflate_layers
        self.grid = inflate_map(self.base, inflate_layers)
        self.temp_blocked = set()                  # dynamic temporary obstacles (cells)

    def to_cell(self, x, y):
        ox, oy = self.origin
        return (int((y - oy) / self.cell), int((x - ox) / self.cell))

    def to_world(self, rc):
        ox, oy = self.origin
        return (ox + (rc[1] + 0.5) * self.cell, oy + (rc[0] + 0.5) * self.cell)

    def _free(self, grid, rc):
        r, c = rc
        return (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]
                and grid[r, c] == 1 and rc not in self.temp_blocked)

    def _nearest_free(self, grid, rc, rad=6):
        if self._free(grid, rc):
            return rc
        for k in range(1, rad + 1):
            for dr in range(-k, k + 1):
                for dc in range(-k, k + 1):
                    cand = (rc[0]+dr, rc[1]+dc)
                    if self._free(grid, cand):
                        return cand
        return None

    def plan(self, start_xy, goal_xy, allow_raw_fallback=True):
        """Trả list waypoint world, hoặc None nếu không có đường (quá hẹp)."""
        for grid in ([self.grid, self.base] if allow_raw_fallback else [self.grid]):
            wp = self._astar(grid, start_xy, goal_xy)
            if wp is not None:
                return wp
        return None

    def _astar(self, grid, start_xy, goal_xy):
        s = self._nearest_free(grid, self.to_cell(*start_xy))
        g = self._nearest_free(grid, self.to_cell(*goal_xy))
        if s is None or g is None:
            return None
        openq = [(0.0, s)]; came = {s: None}; cost = {s: 0.0}
        nbrs = [(1,0,1),(-1,0,1),(0,1,1),(0,-1,1),(1,1,1.41),(1,-1,1.41),(-1,1,1.41),(-1,-1,1.41)]
        while openq:
            _, cur = heapq.heappop(openq)
            if cur == g:
                break
            for dr, dc, w in nbrs:
                nb = (cur[0]+dr, cur[1]+dc)
                if not self._free(grid, nb):
                    continue
                nc = cost[cur] + w
                if nb not in cost or nc < cost[nb]:
                    cost[nb] = nc; came[nb] = cur
                    h = math.hypot(nb[0]-g[0], nb[1]-g[1])
                    heapq.heappush(openq, (nc + h, nb))
        if g not in came:
            return None
        path = []; cur = g
        while cur is not None:
            path.append(self.to_world(cur)); cur = came[cur]
        return path[::-1]

    def block_around(self, xy, radius_cells=1):
        """Đánh dấu vùng quanh `xy` là vật cản tạm (dynamic reroute)."""
        rc = self.to_cell(*xy)
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                self.temp_blocked.add((rc[0]+dr, rc[1]+dc))

    def clear_temp(self):
        self.temp_blocked.clear()


# ====================== Mission Controller =============================== #
class MissionController:
    """Quản lý chuyến: TSP-ish thứ tự gần nhất, hủy trạm không tới được, replan
    động khi kẹt, né người qua SafetyShield. log_fn nhận chuỗi thông báo."""

    def __init__(self, grid, cell, origin, dock, points, inflate_layers=1,
                 log_fn: Optional[Callable[[str], None]] = None,
                 local_policy: Optional[Callable] = None):
        self.planner = AStarPlanner(grid, cell, origin, inflate_layers)
        self.world = self._world_from_grid(grid, cell, origin)
        self.dock = np.array(dock, float)
        self.points = {int(k): np.array(v, float) for k, v in points.items()}
        self.shield = SafetyShield()
        self.log = log_fn or (lambda s: None)
        self.local_policy = local_policy            # optional RL hook (else pure-pursuit)

    @staticmethod
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
        w = World(half_width=cell*2.0, style="mission")
        w.segments = segs
        w.bounds = (ox - cell, ox + C*cell + cell, oy - cell, oy + R*cell + cell)
        return w

    def _order(self, picks):
        """Greedy nearest-neighbour theo độ dài A* (đủ tốt, rẻ)."""
        remaining = list(picks); order = []; cur = tuple(self.dock)
        while remaining:
            best = None; bestlen = 1e18
            for p in remaining:
                wp = self.planner.plan(cur, tuple(self.points[p]))
                L = 1e17 if wp is None else _path_len(wp)
                if L < bestlen:
                    bestlen = L; best = p
            order.append(best); cur = tuple(self.points[best]); remaining.remove(best)
        return order

    def run(self, picks, peds=None, max_steps=25000, lookahead=0.7):
        """Chạy chuyến. Trả dict: trail, route, delivered, cancelled, dodge_pts,
        returned_dock, ped_hits."""
        order = self._order(list(picks))
        pos = self.dock.copy().astype(float); heading = 0.0
        trail = [tuple(pos)]; route = []; dodge_pts = []
        pedtrail = [self._ped_xy(peds)]
        delivered = []; cancelled = []; ped_hits = 0

        queue = list(order) + ["dock"]
        for stop in queue:
            tgt = self.dock if stop == "dock" else self.points[stop]
            wp = self.planner.plan(tuple(pos), tuple(tgt))
            if wp is None:
                if stop != "dock":
                    cancelled.append(int(stop))
                    self.log(f"🔴 BỎ QUA Trạm {stop}: đường quá hẹp/không tới được — "
                             f"Mission Controller tính lại tuyến tới trạm kế!")
                continue
            res = self._drive_leg(pos, heading, wp, tgt, peds, trail, pedtrail,
                                  dodge_pts, route, max_steps, lookahead)
            pos = res["pos"]; heading = res["heading"]; ped_hits += res["ped_hits"]
            if res["arrived"]:
                if stop != "dock":
                    delivered.append(int(stop))
                    self.log(f"🟢 Đã giao thành công tại Trạm {stop}.")
            else:
                if stop != "dock":
                    cancelled.append(int(stop))
                    self.log(f"🔴 BỎ QUA Trạm {stop}: kẹt không tới được — chuyển trạm kế.")
        returned = float(np.linalg.norm(pos - self.dock)) <= 0.6
        if returned:
            self.log("✅ Hoàn thành chuyến đi. Đã về Dock.")
        return {"trail": trail, "pedtrail": pedtrail, "route": route,
                "delivered": delivered, "cancelled": cancelled, "dodge_pts": dodge_pts,
                "returned_dock": returned, "ped_hits": ped_hits,
                "order": [int(x) for x in order], "world": self.world,
                "dock": tuple(self.dock),
                "points": {k: tuple(v) for k, v in self.points.items()}}

    # ---- single leg with pursuit + shield + stuck-replan ---------------- #
    def _drive_leg(self, pos, heading, wp, tgt, peds, trail, pedtrail,
                   dodge_pts, route, max_steps, lookahead):
        path = _densify(wp, 0.2); route.extend(path)
        if not path:                       # tuyến rỗng -> coi như không tới được
            return {"pos": pos, "heading": heading, "ped_hits": 0,
                    "arrived": float(np.linalg.norm(np.array(tgt, float) - pos)) <= 0.5}
        if self.local_policy is not None and hasattr(self.local_policy, "on_leg"):
            self.local_policy.on_leg(path)     # GuidedTracker cần tuyến để tính e_y/e_theta
        tgt = np.array(tgt, float)
        wi = min(1, len(path) - 1); steps = 0; stall = 0; last = pos.copy(); ped_hits = 0
        turn_s = 0.0
        replans = 0
        while np.linalg.norm(tgt - pos) > 0.35 and steps < max_steps:
            steps += 1
            if peds is not None:
                peds.step(DT)
            while wi < len(path)-1 and np.linalg.norm(np.array(path[wi]) - pos) < lookahead:
                wi += 1
            wi = min(wi, len(path) - 1)     # luôn trong phạm vi (tránh IndexError khi path ngắn)
            look = np.array(path[wi])
            # base steering: pure-pursuit (or RL hook)
            if self.local_policy is not None:
                fwd_cmd, turn = self.local_policy(self.world, pos, heading, look)
                fwd = (fwd_cmd * 0.7 + 0.3) * MAX_SPEED
            else:
                bearing = (math.atan2(look[1]-pos[1], look[0]-pos[0]) - heading + math.pi) % (2*math.pi) - math.pi
                turn = float(np.clip(2.2 * bearing, -MAX_TURN, MAX_TURN))
                fwd = MAX_SPEED * (1.0 - 0.4 * min(abs(bearing)/1.2, 1.0))
            # SafetyShield (people only)
            sscale, sturn, status = self.shield.filter(pos, heading, self._ped_xy(peds))
            if status != "clear":
                dodge_pts.append(tuple(pos))
                if status == "slow_sidestep":
                    self.log("⚠️ Đang lách người đi bộ!")
            fwd *= sscale; turn += sturn
            turn_s += 0.25 * (turn - turn_s)               # low-pass -> mượt
            heading = (heading + float(np.clip(turn_s, -MAX_TURN, MAX_TURN)) * DT + math.pi) % (2*math.pi) - math.pi
            npos = pos + np.array([math.cos(heading), math.sin(heading)]) * fwd * DT
            if not self.world.segment_hits_circle(tuple(npos), ROBOT_R) and \
               not (peds is not None and peds.hits_robot(tuple(npos), ROBOT_R)):
                pos = npos
            elif peds is not None and peds.hits_robot(tuple(npos), ROBOT_R):
                ped_hits += 1
            # stuck detector
            if float(np.linalg.norm(pos - last)) < 0.01:
                stall += 1
            else:
                stall = max(0, stall - 2)
            last = pos.copy()
            # (a) corner escape: fan headings toward the next path cell + reverse
            if stall >= 15:
                nxt = np.array(path[min(wi + 1, len(path) - 1)])
                base = math.atan2(nxt[1]-pos[1], nxt[0]-pos[0])
                for off in (0.0, 0.5, -0.5, 1.0, -1.0, 1.6, -1.6, math.pi):
                    cand = pos + np.array([math.cos(base+off), math.sin(base+off)]) * (MAX_SPEED*0.5*DT)
                    if not self.world.segment_hits_circle(tuple(cand), ROBOT_R) and \
                       not (peds is not None and peds.hits_robot(tuple(cand), ROBOT_R)):
                        pos = cand; heading = (base+off+math.pi)%(2*math.pi)-math.pi
                        turn_s = 0.0; wi = min(wi+1, len(path)-1); break
            # (b) persistent stuck -> reroute: block the cell AHEAD (not our own),
            #     then replan from the actual position ("Google Maps reroute").
            if stall >= 45 and replans < 4:
                self.log("🧭 Phát hiện kẹt — đánh dấu vật cản tạm & vẽ lại đường (reroute).")
                ahead = path[min(wi + 2, len(path) - 1)]
                self.planner.block_around(tuple(ahead), radius_cells=1)
                new_wp = self.planner.plan(tuple(pos), tuple(tgt))
                if new_wp is not None and len(_densify(new_wp, 0.2)) > 0:
                    path = _densify(new_wp, 0.2); route.extend(path); wi = min(1, len(path)-1)
                    replans += 1; stall = 0
                else:
                    break                                   # truly unreachable -> cancel
            trail.append(tuple(pos)); pedtrail.append(self._ped_xy(peds))
        self.planner.clear_temp()
        return {"pos": pos, "heading": heading, "ped_hits": ped_hits,
                "arrived": float(np.linalg.norm(tgt - pos)) <= 0.5}

    @staticmethod
    def _ped_xy(peds):
        if peds is None or len(peds.pos) == 0:
            return []
        return [tuple(p) for p in peds.pos]


# ----------------------------- utils -------------------------------------- #
def _path_len(wp):
    a = np.array(wp); return float(np.linalg.norm(np.diff(a, axis=0), axis=1).sum()) if len(a) > 1 else 0.0


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
