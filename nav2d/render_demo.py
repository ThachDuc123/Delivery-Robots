"""GIF rendering for the 3-phase delivery demo (apartment_complex_v1)."""
from __future__ import annotations
import os
import numpy as np


def _save_gif(frames, path, fps=18):
    from PIL import Image
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    dur = max(int(round(1000.0 / fps)), 20)
    pal = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE, colors=64) for f in frames]
    pal[0].save(path, format="GIF", save_all=True, append_images=pal[1:],
                duration=dur, loop=0, optimize=True, disposal=2)
    return path


def gif_phase1_mapping(world, occ_frames, occ, path, fps=18, max_frames=120):
    """Phase 1: occupancy grid filling in as the blind robot explores."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xmin, xmax = occ.xmin, occ.xmin + occ.ncols * occ.cell
    ymin, ymax = occ.ymin, occ.ymin + occ.nrows * occ.cell
    asp = (ymax - ymin) / (xmax - xmin); W = 7.0
    snaps = occ_frames if len(occ_frames) <= max_frames else \
        [occ_frames[i] for i in np.linspace(0, len(occ_frames) - 1, max_frames).astype(int)]
    def img(lo, sn):
        a = np.full(lo.shape, 0.6, np.float32); a[sn & (lo <= 0.3)] = 1.0; a[sn & (lo > 0.3)] = 0.0
        return a
    frames = []
    for (pos, lo, sn) in snaps:
        fig, ax = plt.subplots(figsize=(W, max(3.5, W * asp)))
        ax.imshow(img(lo, sn), cmap="gray", origin="lower", extent=[xmin, xmax, ymin, ymax], vmin=0, vmax=1)
        ax.plot(pos[0], pos[1], "o", color="#1f77b4", ms=7)
        ax.set_title("Phase 1 — robot mù: tự quét & dựng bản đồ (SLAM)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); frames.append(_grab(fig)); plt.close(fig)
    return _save_gif(frames, path, fps)


def gif_delivery(result, path, title, fps=20, max_frames=160):
    """Phase 2/3: robot delivering (with optional moving people)."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mp
    sm = result["saved_map"]; world = result["world"]
    trail = result["trail"]; pedtrail = result.get("pedtrail") or []
    xmin = sm["origin"][0]; ymin = sm["origin"][1]
    g = sm["grid"]; cell = sm["cell"]
    xmax = xmin + g.shape[1] * cell; ymax = ymin + g.shape[0] * cell
    asp = (ymax - ymin) / (xmax - xmin); W = 7.5
    free = [(int(r), int(c)) for r, c in zip(*np.where(g == 1))]
    idx = list(range(len(trail))) if len(trail) <= max_frames else \
        list(np.linspace(0, len(trail) - 1, max_frames).astype(int))
    picks = set(result.get("picks", []))
    frames = []
    for k in idx:
        fig, ax = plt.subplots(figsize=(W, max(3.5, W * asp)))
        for (r, c) in free:
            ax.add_patch(plt.Rectangle((xmin + c*cell, ymin + r*cell), cell, cell,
                                       facecolor="#eef3f7", edgecolor="none"))
        for (x1, y1, x2, y2) in world.segments:
            ax.plot([x1, x2], [y1, y2], color="#333", lw=0.7)
        ax.plot(*sm["dock"], "s", color="#2a8f2a", ms=13)
        for pid, xy in sm["points"].items():
            col = "#d22" if pid in picks else "#bbb"
            ax.plot(*xy, "o", color=col, ms=11 if pid in picks else 6)
            if pid in picks:
                ax.annotate(f"D{pid}", xy, textcoords="offset points", xytext=(0,8),
                            ha="center", weight="bold", color="#d22")
        tr = np.array(trail[:k+1]); ax.plot(tr[:,0], tr[:,1], color="#1f77b4", lw=2, zorder=6)
        ax.add_patch(mp.Circle(trail[k], 0.22, color="#1f3b73", zorder=8))
        if pedtrail and k < len(pedtrail):
            for (px, py) in pedtrail[k]:
                ax.add_patch(mp.Circle((px, py), 0.28, color="#e8902a", zorder=7))
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=10)
        frames.append(_grab(fig)); plt.close(fig)
    return _save_gif(frames, path, fps)


def _grab(fig):
    from PIL import Image
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    return np.asarray(Image.fromarray(buf.reshape(h, w, 4)[:, :, :3]))
