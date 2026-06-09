"""Render a pedestrian-dodging delivery run (Stage 3) as a GIF from captured data.

Reads results/_dodge.json (robot trail + per-step pedestrian positions) and the
map geometry, and animates the robot (blue) weaving past moving people (orange)
on its delivery route. Used to produce the 'successful dodge' demo GIF.
"""
from __future__ import annotations
import json, os, sys
import numpy as np


def render(dodge_json, savepath, fps=20, max_frames=170):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mp
    from PIL import Image
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fixed_maps import build_map

    data = json.load(open(dodge_json))
    m = build_map(data["map"]); w = m["world"]
    trail = data["trail"]; pedtrail = data["pedtrail"]; order = data["order"]
    ped_r = 0.28; rob_r = 0.22
    xmin, xmax, ymin, ymax = w.bounds; asp = (ymax - ymin) / (xmax - xmin); W = 7.5
    n = len(trail)
    idx = list(range(n)) if n <= max_frames else list(np.linspace(0, n - 1, max_frames).astype(int))

    imgs = []
    for k in idx:
        fig, ax = plt.subplots(figsize=(W, max(3.5, W * asp)))
        for (x1, y1, x2, y2) in w.segments:
            ax.plot([x1, x2], [y1, y2], color="#333", lw=1.0, zorder=2)
        ax.plot(*m["dock"], "s", color="#2a8f2a", ms=12, zorder=5)
        for j, pid in enumerate(order):
            xy = m["points"][pid]
            ax.plot(*xy, "o", color="#d22", ms=11, zorder=5)
            ax.annotate(str(j + 1), xy, textcoords="offset points", xytext=(0, 7),
                        ha="center", weight="bold", color="#d22")
        # robot trail + body
        tr = np.array(trail[:k + 1])
        ax.plot(tr[:, 0], tr[:, 1], color="#1f77b4", lw=2, alpha=0.85, zorder=6)
        ax.add_patch(mp.Circle(trail[k], rob_r, color="#1f3b73", zorder=8))
        # pedestrians at this frame
        for (px, py) in pedtrail[k]:
            ax.add_patch(mp.Circle((px, py), ped_r, color="#e8902a", zorder=7))
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("Stage 3: delivery while dodging moving people (LiDAR safety shield)",
                     fontsize=9)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        wpx, hpx = fig.canvas.get_width_height()
        imgs.append(Image.fromarray(buf.reshape(hpx, wpx, 4)[:, :, :3].copy())); plt.close(fig)

    os.makedirs(os.path.dirname(os.path.abspath(savepath)), exist_ok=True)
    dur = max(int(round(1000.0 / fps)), 20)
    pal = [im.convert("P", palette=Image.ADAPTIVE, colors=64) for im in imgs]
    pal[0].save(savepath, format="GIF", save_all=True, append_images=pal[1:],
                duration=dur, loop=0, optimize=True, disposal=2)
    return savepath


if __name__ == "__main__":
    p = render("results/_dodge.json", "results/gifs/dodge_pedestrians.gif")
    print("saved", p, os.path.getsize(p) // 1024, "KB")
