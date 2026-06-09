"""2D top-down rendering + GIF for the Nav2DEnv (matplotlib, light).

Draws the corridor walls, the start (green) and goal (red star), the LiDAR fan,
the robot (blue) and its travelled trail. Used both for single rollouts and to
record GIFs of a trained policy navigating unseen maps.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional

import numpy as np


def _draw_frame(ax, env, lidar=None, title=""):
    import matplotlib.patches as mpatches
    w = env.world
    for (x1, y1, x2, y2) in w.segments:
        ax.plot([x1, x2], [y1, y2], color="#444", lw=1.6, zorder=2)
    # start / goal / current target
    ax.plot(*env.start_pt, "o", color="#2a8f2a", ms=10, zorder=5)
    ax.plot(*env.goal_pt, "*", color="#e23", ms=18, zorder=5)
    tgt = env.goal_pt if env.phase == 0 else env.start_pt
    ax.add_patch(mpatches.Circle(tuple(tgt), env.reach_dist, fill=False,
                                 ec="#e23", ls="--", lw=1, zorder=4))
    # trail
    if len(env.trail) > 1:
        tr = np.array(env.trail)
        ax.plot(tr[:, 0], tr[:, 1], color="#1f77b4", lw=2, alpha=0.8, zorder=4)
    # lidar fan
    if lidar is not None:
        ox, oy = env.pos
        for a, d in zip(env.lidar_angles, lidar * env.lidar_range):
            ang = env.heading + a
            col = "#d33" if d < env.lidar_range * 0.18 else "#9cf"
            ax.plot([ox, ox + math.cos(ang) * d], [oy, oy + math.sin(ang) * d],
                    color=col, lw=0.6, alpha=0.7, zorder=3)
    # robot + heading
    rx, ry = env.pos
    ax.add_patch(mpatches.Circle((rx, ry), env.robot_radius, color="#1f3b73", zorder=6))
    ax.plot([rx, rx + math.cos(env.heading) * env.robot_radius * 2],
            [ry, ry + math.sin(env.heading) * env.robot_radius * 2],
            color="white", lw=2, zorder=7)
    xmin, xmax, ymin, ymax = w.bounds
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal"); ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def record_gif(model, env, filename: str, seed: int = 0, style: Optional[str] = None,
               max_steps: int = 800, fps: int = 18, max_frames: int = 120,
               deterministic: bool = True, title: str = "") -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    obs, info = env.reset(seed=seed, options={"style": style} if style else None)
    frames_state, done, steps = [], False, 0
    reached_goal = round_trip = False
    while not done and steps < max_steps:
        lidar = obs[:env.n_lidar].copy()
        frames_state.append((list(env.trail), tuple(env.pos), env.heading, env.phase, lidar))
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _r, term, trunc, info = env.step(action)
        steps += 1
        reached_goal = reached_goal or info["reached_goal"]
        round_trip = round_trip or info["round_trip"]
        done = term or trunc
    frames_state.append((list(env.trail), tuple(env.pos), env.heading, env.phase,
                         obs[:env.n_lidar].copy()))

    idx = (list(range(len(frames_state))) if len(frames_state) <= max_frames
           else list(np.linspace(0, len(frames_state) - 1, max_frames).astype(int)))
    ttl = title or f"{env.world.style}  ({'round-trip ✓' if round_trip else ('reached ✓' if reached_goal else '...')})"
    imgs = []
    xmin, xmax, ymin, ymax = env.world.bounds
    asp = (ymax - ymin) / (xmax - xmin)
    W = 6.0
    for k in idx:
        trail, pos, head, phase, lidar = frames_state[k]
        fig, ax = plt.subplots(figsize=(W, max(3.0, W * asp)))
        # temporarily set env state for drawing
        env.trail = trail; env.pos = np.array(pos); env.heading = head; env.phase = phase
        _draw_frame(ax, env, lidar=lidar, title=ttl)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w_px, h_px = fig.canvas.get_width_height()
        imgs.append(Image.fromarray(buf.reshape(h_px, w_px, 4)[:, :, :3].copy()))
        plt.close(fig)

    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    dur = max(int(round(1000.0 / fps)), 20)
    pal = [im.convert("P", palette=Image.ADAPTIVE, colors=64) for im in imgs]
    pal[0].save(filename, format="GIF", save_all=True, append_images=pal[1:],
                duration=dur, loop=0, optimize=True, disposal=2)
    return {"gif": filename, "reached_goal": reached_goal, "round_trip": round_trip,
            "steps": steps, "style": env.world.style}


def plot_paths(model, env, styles: List[str], seeds: List[int], savepath: str,
               max_steps: int = 800):
    """Grid of top-down maps with the trained policy's path on several styles."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(styles)
    ncols = min(3, n); nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    summary = []
    for i, style in enumerate(styles):
        ax = axes[i // ncols][i % ncols]
        obs, info = env.reset(seed=seeds[i % len(seeds)], options={"style": style})
        done = False; steps = 0; rg = rt = False; last_lidar = obs[:env.n_lidar]
        while not done and steps < max_steps:
            last_lidar = obs[:env.n_lidar]
            a, _ = model.predict(obs, deterministic=True)
            obs, _r, term, trunc, info = env.step(a)
            steps += 1; rg = rg or info["reached_goal"]; rt = rt or info["round_trip"]
            done = term or trunc
        tag = "round-trip ✓" if rt else ("reached ✓" if rg else "miss")
        _draw_frame(ax, env, lidar=last_lidar, title=f"{style}: {tag} ({steps})")
        summary.append({"style": style, "reached_goal": rg, "round_trip": rt, "steps": steps})
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle("PPO sensor-only navigation — paths on unseen maps", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(savepath)), exist_ok=True)
    fig.savefig(savepath, dpi=110, bbox_inches="tight")
    return summary
