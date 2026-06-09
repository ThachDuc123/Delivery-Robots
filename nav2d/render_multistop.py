"""GIF of the multi-stop LSTM policy doing a full delivery trip on a fixed map.

Runs the policy directly in MultiStopEnv (the deploy loop), records the trail,
and draws the map + planned route + dock + chosen stops (numbered in visit order)
+ the robot. The robot visibly turns around at each stop (grace period) and heads
to the next, finishing back at the dock.
"""
from __future__ import annotations
import math, os
from typing import List, Optional
import numpy as np


def record_trip_gif(model, norm, env, savepath, seed=0, options=None,
                    fps=20, max_frames=200, title=""):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mp
    from PIL import Image

    o, info = env.reset(seed=seed, options=options)
    order = info["order"]
    state = None; es = np.ones(1, bool); done = False
    trail = [tuple(env.pos)]; stop_pts = []
    while not done:
        a, state = model.predict(norm(o)[None], state=state, episode_start=es, deterministic=True)
        es = np.zeros(1, bool)
        o, r, t, tr, inf = env.step(a[0]); trail.append(tuple(env.pos))
        if inf["stop"]:
            stop_pts.append(tuple(env.pos))
        done = t or tr
    dock_ok = inf["arrived_dock"]

    m = env.map; w = env.world
    idx = (list(range(len(trail))) if len(trail) <= max_frames
           else list(np.linspace(0, len(trail) - 1, max_frames).astype(int)))
    xmin, xmax, ymin, ymax = w.bounds; asp = (ymax - ymin) / (xmax - xmin); W = 8.0
    order_str = "DOCK -> " + " -> ".join(f"#{p}" for p in order) + " -> DOCK"
    ttl = title or f"deliver {list(order)}  ({'returned to dock OK' if dock_ok else 'incomplete'})"

    frames = []
    for k in idx:
        fig, ax = plt.subplots(figsize=(W, max(3.5, W * asp)))
        for (x1, y1, x2, y2) in w.segments:
            ax.plot([x1, x2], [y1, y2], color="#333", lw=1.0, zorder=2)
        # all points grey, chosen red+order, dock green
        oi = {p: i for i, p in enumerate(order)}
        for pid, xy in m["points"].items():
            if pid in order:
                ax.plot(*xy, "o", color="#d22", ms=12, zorder=6)
                ax.annotate(f"{oi[pid]+1}", xy, textcoords="offset points", xytext=(0, 7),
                            ha="center", fontsize=10, weight="bold", color="#d22")
            else:
                ax.plot(*xy, "o", color="#ccc", ms=6, zorder=4)
        ax.plot(*m["dock"], "s", color="#2a8f2a", ms=14, zorder=6)
        tr = np.array(trail[:k + 1])
        ax.plot(tr[:, 0], tr[:, 1], color="#1f77b4", lw=2.2, alpha=0.9, zorder=5)
        rx, ry = trail[k]
        ax.add_patch(mp.Circle((rx, ry), env.robot_radius, color="#1f3b73", zorder=8))
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{ttl}\n{order_str}", fontsize=9)
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
    return {"gif": savepath, "dock": dock_ok, "order": list(order), "steps": len(trail)}
