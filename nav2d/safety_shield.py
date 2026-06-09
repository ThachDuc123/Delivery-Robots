"""BƯỚC 4 — Safety Shield (tách biệt né ĐỘNG).

Shield CHỈ nhận diện NGƯỜI (pedestrians) — tuyệt đối không can thiệp vào tường
tĩnh (đã có A* + RL lo). Cơ chế override điều khiển:
  * người cách < react (1.0 m) trong cone phía trước  -> LÁCH, lực bẻ lái tỉ lệ
    NGHỊCH khoảng cách (càng gần lách càng mạnh), giảm tốc.
  * người cách < brake (0.5 m)                         -> PHANH GẤP (v -> ~0).
  * không có người gần                                 -> trả quyền cho RL.
"""

from __future__ import annotations

import math
import numpy as np


class SafetyShield:
    def __init__(self, react=1.0, brake=0.5, cone_deg=35.0, robot_r=0.22, ped_r=0.28):
        self.react = react
        self.brake = brake
        self.cone = math.radians(cone_deg)
        self.robot_r = robot_r
        self.ped_r = ped_r

    def filter(self, pos, heading, ped_positions):
        """Return (speed_scale, extra_turn, status).
        speed_scale in [0,1]; extra_turn in rad/s to ADD to the RL turn;
        status in {clear, slow_sidestep, brake}."""
        if ped_positions is None or len(ped_positions) == 0:
            return 1.0, 0.0, "clear"
        pos = np.asarray(pos, float)
        fwd = np.array([math.cos(heading), math.sin(heading)])
        left = np.array([-fwd[1], fwd[0]])
        worst = None
        for p in ped_positions:
            rel = np.asarray(p, float) - pos
            d = float(np.linalg.norm(rel)) - self.ped_r        # surface distance
            if d > self.react:
                continue
            ahead = float(rel @ fwd)
            ang = abs(math.atan2(float(rel @ left), ahead)) if ahead != 0 else math.pi
            if ahead > 0 and ang < self.cone and (worst is None or d < worst[0]):
                worst = (d, float(rel @ left))
        if worst is None:
            return 1.0, 0.0, "clear"
        d, side = worst
        # closeness 0 (at react) .. 1 (at brake) -> steer/​slow proportional
        closeness = float(np.clip(1.0 - (d - self.brake) / max(self.react - self.brake, 1e-6), 0.0, 1.0))
        extra_turn = -math.copysign(1.0, side or 1.0) * closeness * 1.4   # rad/s, away from person
        if d < self.brake:
            return 0.0, extra_turn, "brake"
        return (1.0 - 0.7 * closeness), extra_turn, "slow_sidestep"
