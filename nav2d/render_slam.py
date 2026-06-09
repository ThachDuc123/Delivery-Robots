"""Render the SLAM->delivery pipeline as a GIF.

Two visual phases in one animation:
  Phase 1 (mapping): the occupancy grid fills in live (grey=unknown, white=free,
    black=occupied) as the robot drives the frontier-exploration path.
  Phase 2 (delivery): on the discovered map, the robot drives the TSP route to
    each delivery point and back to the dock.
"""
from __future__ import annotations
import os
import numpy as np


def record_slam_gif(result, savepath, fps=18, max_frames=160):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    occ = result["occ"]
    frames_meta = result["frames"]            # (pos, logodds, seen) snapshots
    ex_trail = result["explore_trail"]
    dv_trail = result["deliver_trail"]
    xmin, xmax = occ.xmin, occ.xmin + occ.ncols * occ.cell
    ymin, ymax = occ.ymin, occ.ymin + occ.nrows * occ.cell
    asp = (ymax - ymin) / (xmax - xmin); W = 7.0

    def grid_img(logodds, seen):
        img = np.full(logodds.shape, 0.6, dtype=np.float32)   # unknown=grey
        img[seen & (logodds <= 0.3)] = 1.0                    # free=white
        img[seen & (logodds > 0.3)] = 0.0                     # occ=black
        return img

    # build frame list: sampled mapping snapshots, then delivery frames on final map
    map_snaps = frames_meta if len(frames_meta) <= max_frames // 2 else \
        [frames_meta[i] for i in np.linspace(0, len(frames_meta) - 1, max_frames // 2).astype(int)]
    final_img = grid_img(occ.logodds, occ.seen)
    dv_idx = list(range(len(dv_trail))) if len(dv_trail) <= max_frames // 2 else \
        list(np.linspace(0, len(dv_trail) - 1, max_frames // 2).astype(int))

    imgs = []
    # phase 1
    for (pos, lo, sn) in map_snaps:
        fig, ax = plt.subplots(figsize=(W, max(3.5, W * asp)))
        ax.imshow(grid_img(lo, sn), cmap="gray", origin="lower",
                  extent=[xmin, xmax, ymin, ymax], vmin=0, vmax=1)
        ax.plot(pos[0], pos[1], "o", color="#1f77b4", ms=7)
        ax.set_title("Phase 1: SLAM blind mapping (frontier exploration)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        imgs.append(_grab(fig)); plt.close(fig)
    # phase 2
    pts = result.get("_points_xy", None)
    for k in dv_idx:
        fig, ax = plt.subplots(figsize=(W, max(3.5, W * asp)))
        ax.imshow(final_img, cmap="gray", origin="lower",
                  extent=[xmin, xmax, ymin, ymax], vmin=0, vmax=1)
        tr = np.array(dv_trail[:k + 1])
        ax.plot(tr[:, 0], tr[:, 1], color="#d22", lw=2, zorder=5)
        ax.plot(dv_trail[k][0], dv_trail[k][1], "o", color="#1f3b73", ms=8, zorder=6)
        ax.set_title(f"Phase 2: deliver on discovered map  (order {result['order']})",
                     fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        imgs.append(_grab(fig)); plt.close(fig)

    os.makedirs(os.path.dirname(os.path.abspath(savepath)), exist_ok=True)
    dur = max(int(round(1000.0 / fps)), 20)
    pal = [im.convert("P", palette=Image.ADAPTIVE, colors=64) for im in imgs]
    pal[0].save(savepath, format="GIF", save_all=True, append_images=pal[1:],
                duration=dur, loop=0, optimize=True, disposal=2)
    return savepath


def _grab(fig):
    from PIL import Image
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    return Image.fromarray(buf.reshape(h, w, 4)[:, :, :3].copy())
