"""Render multi-stop delivery on a fixed map: static plan image + animated GIF.

Draws the map walls, the dock, all delivery points (chosen ones highlighted with
their visit order), the planned global route, and the robot's actual trail. The
GIF animates the robot delivering to each chosen point in the TSP-optimised order
and returning to the dock.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional

import numpy as np


def _draw_base(ax, runner, plan, chosen):
    import matplotlib.patches as mpatches
    w = runner.world
    for (x1, y1, x2, y2) in w.segments:
        ax.plot([x1, x2], [y1, y2], color="#333", lw=1.0, zorder=2)
    # all points (grey), chosen (red w/ order), dock (green square)
    order = plan["order"]
    order_idx = {pid: i for i, pid in enumerate(order)}
    for pid, xy in runner.map["points"].items():
        if pid in chosen:
            ax.plot(*xy, "o", color="#d22", ms=12, zorder=6)
            ax.annotate(f"{order_idx[pid]+1}·#{pid}", xy, textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=9, weight="bold", color="#d22")
        else:
            ax.plot(*xy, "o", color="#bbb", ms=7, zorder=4)
            ax.annotate(str(pid), xy, textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=7, color="#999")
    ax.plot(*runner.map["dock"], "s", color="#2a8f2a", ms=14, zorder=6)
    ax.annotate("DOCK", runner.map["dock"], textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=9, weight="bold", color="#176")
    # planned global route (light)
    wps = np.array(plan["waypoints"])
    ax.plot(wps[:, 0], wps[:, 1], "--", color="#69c", lw=1.2, alpha=0.7, zorder=3)
    xmin, xmax, ymin, ymax = w.bounds
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def plot_plan(runner, plan, chosen, savepath, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 7))
    _draw_base(ax, runner, plan, chosen)
    order = " → ".join(["DOCK"] + [f"#{p}" for p in plan["order"]] + ["DOCK"])
    ax.set_title(f"{title}\nroute: {order}   |   {plan['total_dist']:.1f} m   |   "
                 f"battery ~{plan['battery_pct']:.0f}%", fontsize=10)
    os.makedirs(os.path.dirname(os.path.abspath(savepath)), exist_ok=True)
    fig.savefig(savepath, dpi=110, bbox_inches="tight"); plt.close(fig)


def record_delivery_gif(runner, chosen: List[int], savepath: str, seed: int = 0,
                        fps: int = 20, max_frames: int = 160, title="") -> Dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    res = runner.run(chosen, seed=seed)
    plan = res["plan"]
    trail = res.get("trail", [])
    if not trail:
        return res
    idx = (list(range(len(trail))) if len(trail) <= max_frames
           else list(np.linspace(0, len(trail) - 1, max_frames).astype(int)))
    w = runner.world
    xmin, xmax, ymin, ymax = w.bounds
    asp = (ymax - ymin) / (xmax - xmin); W = 8.0
    order = " → ".join(["DOCK"] + [f"#{p}" for p in plan["order"]] + ["DOCK"])
    base_title = title or f"deliver {plan['order']}  ({plan['total_dist']:.1f} m, ~{plan['battery_pct']:.0f}% batt)"

    frames = []
    for k in idx:
        fig, ax = plt.subplots(figsize=(W, max(3.5, W * asp)))
        _draw_base(ax, runner, plan, chosen)
        tr = np.array(trail[:k + 1])
        ax.plot(tr[:, 0], tr[:, 1], color="#1f77b4", lw=2.2, alpha=0.9, zorder=5)
        rx, ry = trail[k]
        import matplotlib.patches as mpatches
        ax.add_patch(mpatches.Circle((rx, ry), runner.env.robot_radius, color="#1f3b73", zorder=8))
        # progress: how many delivered by this frame (approx by nearest point passed)
        ax.set_title(f"{base_title}\n{order}", fontsize=9)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        wpx, hpx = fig.canvas.get_width_height()
        frames.append(Image.fromarray(buf.reshape(hpx, wpx, 4)[:, :, :3].copy()))
        plt.close(fig)

    os.makedirs(os.path.dirname(os.path.abspath(savepath)), exist_ok=True)
    dur = max(int(round(1000.0 / fps)), 20)
    pal = [im.convert("P", palette=Image.ADAPTIVE, colors=64) for im in frames]
    pal[0].save(savepath, format="GIF", save_all=True, append_images=pal[1:],
                duration=dur, loop=0, optimize=True, disposal=2)
    res["gif"] = savepath
    return res
