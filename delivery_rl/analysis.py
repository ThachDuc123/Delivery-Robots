"""2D top-down map review + comparison charts for PPO / SAC / TD3.

All plotting is matplotlib (static images -> light, never freezes the notebook,
unlike the looping GIFs). Two groups of helpers:

  * ``rollout_trajectory`` + ``plot_scenarios_grid`` : run each model on the
    SAME map and SAME destination locker and draw the top-down corridor with the
    three robot paths overlaid -- so you can visually compare how PPO/SAC/TD3
    drive to different lockers across different maps/obstacle layouts.
  * ``benchmark`` + ``plot_benchmark_charts`` : evaluate the three models over
    many episodes and draw grouped bar charts (delivery %, reward, steps,
    collisions) plus a path-length comparison.

Coordinates are world metres; the corridor backbone runs along x, width along y.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from delivery_rl.configs.loader import default_config_path, load_config
from delivery_rl.envs.corridor_delivery_env import CorridorDeliveryEnv

ALGO_COLORS = {"ppo": "#1f77b4", "sac": "#ff7f0e", "td3": "#2ca02c"}


# --------------------------------------------------------------------------- #
#  Rollout that records the robot's (x, y) path + the static scene geometry
# --------------------------------------------------------------------------- #
def rollout_trajectory(model, level: int, override: dict, seed: int,
                       max_steps: int = 900) -> dict:
    cfg = load_config(default_config_path())
    cfg["env"]["curriculum"]["level"] = level
    cfg["env"]["max_episode_steps"] = max_steps
    if override:
        cfg["env"]["scenario_override"] = override

    env = CorridorDeliveryEnv(config=cfg)
    obs, info = env.reset(seed=seed)
    n = info["num_parcels"]
    xs, ys = [], []
    done, steps, reached = False, 0, False
    while not done and steps < max_steps:
        x, y, _ = env.robot.get_pose()
        xs.append(x); ys.append(y)
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, term, trunc, info = env.step(action)
        steps += 1
        if info["deliveries_done"] >= n:
            reached = True
            x, y, _ = env.robot.get_pose()
            xs.append(x); ys.append(y)
            break
        done = term or trunc

    # capture geometry for drawing (same for all models on this episode)
    scene = env.scene
    geom = {
        "length": scene.corridor_length,
        "width": scene.corridor_width,
        "lockers": [(l.id, l.pos[0], l.pos[1], l.dock[0], l.dock[1]) for l in scene.lockers],
        "obstacles": list(env.world.obstacles_geom),
        "dock": (scene.dock_pos[0], scene.dock_pos[1]),
        "start": (xs[0], ys[0]) if xs else (0.0, 0.0),
        "target": None,
        "locker_size": cfg["env"]["world"]["locker_size"],
        "walls": list(getattr(env.world, "draw_rects", [])),   # (cx,cy,hx,hy,kind)
        "bounds": tuple(scene.bounds),
    }
    _, tgt = env.task.current_target_xy(np.array(geom["start"], dtype=np.float32))
    # target locker (the one actually assigned this episode)
    if env.task.manifest:
        lid = env.task.manifest[0].locker_id
        lk = env.task.locker_by_id(lid)
        geom["target"] = (lid, lk.pos[0], lk.pos[1], lk.dock[0], lk.dock[1])
    env.close()

    path = np.array(list(zip(xs, ys)), dtype=np.float32) if xs else np.zeros((1, 2), np.float32)
    length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))) if len(path) > 1 else 0.0
    return {"path": path, "geom": geom, "reached": reached, "steps": steps, "path_len": length}


# --------------------------------------------------------------------------- #
#  Draw one top-down map with the three model paths overlaid
# --------------------------------------------------------------------------- #
def _draw_map(ax, geom: dict, title: str) -> None:
    import matplotlib.patches as mpatches

    L, W = geom["length"], geom["width"]
    hx, hy = L / 2.0, W / 2.0
    xmin, xmax, ymin, ymax = geom.get("bounds", (-hx - 1.5, hx + 1.5, -hy - 3, hy + 3))
    # whole floor area (rooms + arc included)
    ax.add_patch(mpatches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    facecolor="#fafafa", edgecolor="none", zorder=0))
    # actual walls / rooms / arc segments (so different maps look different)
    wall_cols = {"wall": "#7d8390", "room": "#8fa888", "elevator": "#5577aa",
                 "stair": "#aaaaaa"}
    for (cx, cy, whx, why, kind) in geom.get("walls", []):
        ax.add_patch(mpatches.Rectangle((cx - whx, cy - why), 2 * whx, 2 * why,
                                        facecolor=wall_cols.get(kind, "#7d8390"),
                                        edgecolor="none", zorder=1))
    lw, ldp, _ = geom["locker_size"]
    # lockers (flush to walls) -- grey boxes; target gets a red dot
    for lid, lx, ly, dx, dy in geom["lockers"]:
        ax.add_patch(mpatches.Rectangle((lx - lw / 2, ly - ldp / 2), lw, ldp,
                                        facecolor="#c9b48f", edgecolor="#8a7654",
                                        linewidth=0.5, zorder=1))
    # static obstacles
    for ox, oy, s in geom.get("obstacles", []):
        ax.add_patch(mpatches.Rectangle((ox - s, oy - s), 2 * s, 2 * s,
                                        facecolor="#d9534f", edgecolor="#a33",
                                        linewidth=0.5, zorder=2))
    # dock (charging) + start
    dkx, dky = geom["dock"]
    ax.plot(dkx, dky, marker="s", color="#2a8", ms=9, zorder=4)
    ax.annotate("dock", (dkx, dky), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=8, color="#176")
    sx, sy = geom["start"]
    ax.plot(sx, sy, marker="o", color="#333", ms=7, zorder=4)
    # target locker red dot
    if geom.get("target"):
        lid, tx, ty, dx, dy = geom["target"]
        ax.plot(tx, ty, marker="o", color="red", ms=12, zorder=6)
        ax.annotate(f"locker #{lid}", (tx, ty), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8, color="red", weight="bold")
        ax.plot(dx, dy, marker="*", color="red", ms=11, zorder=6)  # dock point

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")


def plot_scenarios_grid(models: Dict, scenarios: List[dict], seed: int = 2024,
                        savepath: Optional[str] = None):
    """For each scenario draw one top-down map with PPO/SAC/TD3 paths overlaid.

    ``scenarios`` is a list of dicts: {name, level, override}. ``override`` may
    include ``force_locker_id`` so every model targets the same locker."""
    import matplotlib.pyplot as plt

    n = len(scenarios)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 3.0 * nrows),
                             squeeze=False)
    summary = []
    for idx, sc in enumerate(scenarios):
        ax = axes[idx // ncols][idx % ncols]
        geom_ref = None
        for algo, model in models.items():
            roll = rollout_trajectory(model, sc["level"], sc.get("override", {}), seed)
            if geom_ref is None:
                geom_ref = roll["geom"]
            path = roll["path"]
            ax.plot(path[:, 0], path[:, 1], color=ALGO_COLORS.get(algo, "k"),
                    lw=2.0, alpha=0.85,
                    label=f"{algo.upper()} ({'reach' if roll['reached'] else 'miss'}, "
                          f"{roll['path_len']:.1f}m)", zorder=5)
            summary.append({"scenario": sc["name"], "algo": algo,
                            "reached": roll["reached"], "steps": roll["steps"],
                            "path_len": roll["path_len"]})
        _draw_map(ax, geom_ref, sc["name"])
        ax.legend(fontsize=7, loc="upper center", ncol=3)
    # hide any unused axes
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle("Top-down map review — PPO vs SAC vs TD3 paths to the red-dot locker",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=110, bbox_inches="tight")
    return fig, summary


# --------------------------------------------------------------------------- #
#  Quantitative benchmark + comparison charts
# --------------------------------------------------------------------------- #
def benchmark(models: Dict, level: int = 0, override: Optional[dict] = None,
              episodes: int = 12, seed0: int = 300, max_steps: int = 900) -> Dict:
    results = {}
    for algo, model in models.items():
        R, S, D, C, OK, PL = [], [], [], [], [], []
        for ep in range(episodes):
            cfg = load_config(default_config_path())
            cfg["env"]["curriculum"]["level"] = level
            cfg["env"]["max_episode_steps"] = max_steps
            if override:
                cfg["env"]["scenario_override"] = override
            env = CorridorDeliveryEnv(config=cfg)
            obs, info = env.reset(seed=seed0 + ep)
            n = info["num_parcels"]
            done = False; tot = steps = coll = dd = 0
            px, py, _ = env.robot.get_pose(); plen = 0.0
            while not done:
                a, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(a)
                tot += r; steps += 1; coll += int(info["collision"]); dd = info["deliveries_done"]
                nx, ny, _ = env.robot.get_pose()
                plen += float(np.hypot(nx - px, ny - py)); px, py = nx, ny
                done = term or trunc
            env.close()
            R.append(tot); S.append(steps); D.append(dd / max(n, 1))
            C.append(coll); OK.append(int(dd >= n)); PL.append(plen)
        results[algo] = {
            "avg_reward": float(np.mean(R)), "avg_steps": float(np.mean(S)),
            "delivery_rate": float(np.mean(D)), "reach_rate": float(np.mean(OK)),
            "collisions": float(np.mean(C)), "avg_path_len": float(np.mean(PL)),
            "reward_std": float(np.std(R)),
        }
    return results


def plot_benchmark_charts(results: Dict, title_suffix: str = "",
                          savepath: Optional[str] = None):
    import matplotlib.pyplot as plt

    algos = list(results.keys())
    colors = [ALGO_COLORS.get(a, "gray") for a in algos]
    panels = [
        ("delivery_rate", "Delivery rate", lambda v: v * 100, "%"),
        ("avg_reward", "Avg reward", lambda v: v, ""),
        ("avg_steps", "Avg steps", lambda v: v, ""),
        ("collisions", "Collisions / ep", lambda v: v, ""),
        ("avg_path_len", "Avg path length", lambda v: v, "m"),
        ("reach_rate", "Reach rate", lambda v: v * 100, "%"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (key, label, fn, unit) in zip(axes.ravel(), panels):
        vals = [fn(results[a][key]) for a in algos]
        bars = ax.bar([a.upper() for a in algos], vals, color=colors, alpha=0.85)
        ax.set_title(label, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        if unit == "%":
            ax.set_ylim(0, 105)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{v:.1f}{unit}", ha="center", va="bottom", fontsize=9)
    fig.suptitle(f"PPO vs SAC vs TD3 — quantitative comparison {title_suffix}",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=110, bbox_inches="tight")
    return fig


def pick_common_lockers(models: Dict, level: int, n_lockers: int = 3,
                        seed: int = 2024) -> List[int]:
    """Return locker ids that ALL models can reach (for fair side-by-side maps)."""
    cfg = load_config(default_config_path()); cfg["env"]["curriculum"]["level"] = level
    env = CorridorDeliveryEnv(config=cfg); env.reset(seed=seed)
    ids = [l.id for l in env.scene.lockers]
    env.close()
    good = []
    for lid in ids:
        ok = True
        for model in models.values():
            roll = rollout_trajectory(model, level, {"force_locker_id": lid}, seed, max_steps=700)
            if not roll["reached"]:
                ok = False
                break
        if ok:
            good.append(lid)
        if len(good) >= n_lockers:
            break
    return good


# --------------------------------------------------------------------------- #
#  Interactive 2D: pick a locker -> animated top-down GIF of the robot driving
# --------------------------------------------------------------------------- #
def list_lockers(seed: int = 0) -> List[dict]:
    """List every locker with its id, area (side) and (x, y) -- so you know which
    id to pass to :func:`animate_2d_rollout`."""
    cfg = load_config(default_config_path())
    env = CorridorDeliveryEnv(config=cfg)
    env.reset(seed=seed)
    dock = np.array(env.scene.dock_pos[:2], dtype=np.float32)
    out = []
    for l in env.scene.lockers:
        out.append({"id": l.id, "area": l.side,
                    "pos": (round(l.pos[0], 1), round(l.pos[1], 1)),
                    "dist_from_dock": round(float(np.linalg.norm(np.array(l.dock) - dock)), 1)})
    env.close()
    return out


def animate_2d_rollout(model, locker_id: int, filename: str = "rollout_2d.gif",
                       level: int = 1, seed: int = 2024, max_steps: int = 900,
                       num_obstacles: int = 0, num_pedestrians: int = 0,
                       fps: int = 15, max_frames: int = 120,
                       label: str = "", title: Optional[str] = None) -> dict:
    """Run ONE model to a CHOSEN locker and save a 2D top-down animation (GIF).

    This is the lightweight 2D equivalent of the 3D GIF gallery: the robot, its
    travelled path, the LiDAR-free top-down map (walls / rooms / arc), and the
    red-dot destination are drawn with matplotlib and written as an animated GIF.

    Returns a dict with reached/steps/path_len and the gif path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    override = {"force_locker_id": int(locker_id)}
    if num_obstacles:
        override["num_obstacles"] = int(num_obstacles)
    if num_pedestrians:
        override["num_pedestrians"] = int(num_pedestrians)

    cfg = load_config(default_config_path())
    cfg["env"]["curriculum"]["level"] = level
    cfg["env"]["max_episode_steps"] = max_steps
    cfg["env"]["scenario_override"] = override

    env = CorridorDeliveryEnv(config=cfg)
    obs, info = env.reset(seed=seed)
    n = info["num_parcels"]

    # record path + heading, and the moving pedestrian positions per step
    xs, ys, yaws, peds_seq = [], [], [], []
    done, steps, reached = False, 0, False
    while not done and steps < max_steps:
        x, y, yaw = env.robot.get_pose()
        xs.append(x); ys.append(y); yaws.append(yaw)
        peds_seq.append([tuple(s.pos) for s in env.pedestrians.get_states((x, y))]
                        if env.pedestrians.body_ids else [])
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, term, trunc, info = env.step(action)
        steps += 1
        if info["deliveries_done"] >= n:
            reached = True
            x, y, yaw = env.robot.get_pose()
            xs.append(x); ys.append(y); yaws.append(yaw); peds_seq.append([])
            break
        done = term or trunc

    scene = env.scene
    geom = {
        "length": scene.corridor_length, "width": scene.corridor_width,
        "lockers": [(l.id, l.pos[0], l.pos[1], l.dock[0], l.dock[1]) for l in scene.lockers],
        "obstacles": list(env.world.obstacles_geom),
        "dock": (scene.dock_pos[0], scene.dock_pos[1]),
        "start": (xs[0], ys[0]) if xs else (0.0, 0.0),
        "target": None, "locker_size": cfg["env"]["world"]["locker_size"],
        "walls": list(getattr(env.world, "draw_rects", [])), "bounds": tuple(scene.bounds),
    }
    lk = env.task.locker_by_id(int(locker_id))
    geom["target"] = (lk.id, lk.pos[0], lk.pos[1], lk.dock[0], lk.dock[1])
    ped_r = env.pedestrians.radius if env.pedestrians.body_ids else 0.25
    rob_r = max(env.robot.base_size[0], env.robot.base_size[1]) / 2.0
    env.close()

    # subsample frames to keep the GIF light
    nf = len(xs)
    idx = list(range(nf)) if nf <= max_frames else \
        list(np.linspace(0, nf - 1, max_frames).astype(int))

    xmin, xmax, ymin, ymax = geom["bounds"]
    width_in = 11.0
    height_in = max(3.0, width_in * (ymax - ymin) / (xmax - xmin))
    frames = []
    ttl = title or (f"{label or 'robot'} -> locker #{locker_id} "
                    f"({'reached' if reached else 'did not reach'})")
    for k in idx:
        fig, ax = plt.subplots(figsize=(width_in, height_in))
        _draw_map(ax, geom, ttl)
        # travelled path up to this frame
        ax.plot(xs[:k + 1], ys[:k + 1], color="#1f77b4", lw=2.0, alpha=0.9, zorder=5)
        # pedestrians at this frame
        for (px, py) in peds_seq[k]:
            ax.add_patch(plt.Circle((px, py), ped_r, color="#e8a33d", zorder=6))
        # robot body + heading arrow
        rx, ry, ryaw = xs[k], ys[k], yaws[k]
        ax.add_patch(plt.Circle((rx, ry), rob_r, color="#1f3b73", zorder=7))
        ax.plot([rx, rx + math.cos(ryaw) * rob_r * 1.8],
                [ry, ry + math.sin(ryaw) * rob_r * 1.8], color="white", lw=2, zorder=8)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        frames.append(Image.fromarray(buf.reshape(h, w, 4)[:, :, :3].copy()))
        plt.close(fig)

    if frames:
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        dur = max(int(round(1000.0 / float(fps))), 20)
        pal = [im.convert("P", palette=Image.ADAPTIVE, colors=64) for im in frames]
        pal[0].save(filename, format="GIF", save_all=True, append_images=pal[1:],
                    duration=dur, loop=0, optimize=True, disposal=2)

    path = np.array(list(zip(xs, ys)), dtype=np.float32) if xs else np.zeros((1, 2), np.float32)
    plen = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))) if len(path) > 1 else 0.0
    return {"gif": filename if frames else None, "reached": reached, "steps": steps,
            "path_len": plen, "frames": len(frames), "locker_id": int(locker_id)}


def _run_one(model, locker_id, level, seed, max_steps, override):
    """Roll one model to a locker; return per-step (x, y, yaw) + reached + geom."""
    cfg = load_config(default_config_path())
    cfg["env"]["curriculum"]["level"] = level
    cfg["env"]["max_episode_steps"] = max_steps
    ov = {"force_locker_id": int(locker_id)}
    ov.update(override or {})
    cfg["env"]["scenario_override"] = ov
    env = CorridorDeliveryEnv(config=cfg)
    obs, info = env.reset(seed=seed)
    n = info["num_parcels"]
    xs, ys, yaws, peds_seq = [], [], [], []
    done, steps, reached = False, 0, False
    while not done and steps < max_steps:
        x, y, yaw = env.robot.get_pose()
        xs.append(x); ys.append(y); yaws.append(yaw)
        peds_seq.append([tuple(s.pos) for s in env.pedestrians.get_states((x, y))]
                        if env.pedestrians.body_ids else [])
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, term, trunc, info = env.step(action)
        steps += 1
        if info["deliveries_done"] >= n:
            reached = True
            x, y, yaw = env.robot.get_pose()
            xs.append(x); ys.append(y); yaws.append(yaw); peds_seq.append(peds_seq[-1] if peds_seq else [])
            break
        done = term or trunc
    scene = env.scene
    geom = {
        "length": scene.corridor_length, "width": scene.corridor_width,
        "lockers": [(l.id, l.pos[0], l.pos[1], l.dock[0], l.dock[1]) for l in scene.lockers],
        "obstacles": list(env.world.obstacles_geom),
        "dock": (scene.dock_pos[0], scene.dock_pos[1]),
        "start": (xs[0], ys[0]) if xs else (0.0, 0.0),
        "target": None, "locker_size": cfg["env"]["world"]["locker_size"],
        "walls": list(getattr(env.world, "draw_rects", [])), "bounds": tuple(scene.bounds),
    }
    lk = env.task.locker_by_id(int(locker_id))
    geom["target"] = (lk.id, lk.pos[0], lk.pos[1], lk.dock[0], lk.dock[1])
    rob_r = max(env.robot.base_size[0], env.robot.base_size[1]) / 2.0
    ped_r = env.pedestrians.radius if env.pedestrians.body_ids else 0.25
    env.close()
    return {"xs": xs, "ys": ys, "yaws": yaws, "peds": peds_seq, "reached": reached,
            "steps": steps, "geom": geom, "rob_r": rob_r, "ped_r": ped_r}


def animate_2d_compare(models: Dict, locker_id: int, filename: str = "compare_2d.gif",
                       level: int = 1, seed: int = 2024, max_steps: int = 900,
                       num_obstacles: int = 0, num_pedestrians: int = 0,
                       fps: int = 15, max_frames: int = 120) -> dict:
    """Run ALL given models to the SAME chosen locker and render ONE 2D GIF with
    every robot moving at once (PPO/SAC/TD3 each in its own colour).

    Each model runs on its own env instance (same seed/map/destination), and the
    per-step poses are overlaid frame-by-frame so you can watch the three race to
    the same red-dot locker side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    override = {}
    if num_obstacles:
        override["num_obstacles"] = int(num_obstacles)
    if num_pedestrians:
        override["num_pedestrians"] = int(num_pedestrians)

    runs = {a: _run_one(m, locker_id, level, seed, max_steps, override)
            for a, m in models.items()}
    geom = next(iter(runs.values()))["geom"]          # shared map geometry
    nf = max(len(r["xs"]) for r in runs.values())
    idx = list(range(nf)) if nf <= max_frames else \
        list(np.linspace(0, nf - 1, max_frames).astype(int))

    xmin, xmax, ymin, ymax = geom["bounds"]
    width_in = 11.0
    height_in = max(3.0, width_in * (ymax - ymin) / (xmax - xmin))
    summary = {a: {"reached": r["reached"], "steps": r["steps"]} for a, r in runs.items()}
    title = "PPO vs SAC vs TD3 -> locker #%d  (" % locker_id + \
            ", ".join(f"{a.upper()}:{'✓' if s['reached'] else '✗'}" for a, s in summary.items()) + ")"

    frames = []
    for k in idx:
        fig, ax = plt.subplots(figsize=(width_in, height_in))
        _draw_map(ax, geom, title)
        # pedestrians (from the first run that has them)
        for r in runs.values():
            if r["peds"]:
                kk = min(k, len(r["peds"]) - 1)
                for (px, py) in r["peds"][kk]:
                    ax.add_patch(plt.Circle((px, py), r["ped_r"], color="#e8a33d", zorder=6))
                break
        for algo, r in runs.items():
            col = ALGO_COLORS.get(algo, "#333")
            kk = min(k, len(r["xs"]) - 1)
            ax.plot(r["xs"][:kk + 1], r["ys"][:kk + 1], color=col, lw=2.0, alpha=0.85,
                    zorder=5, label=f"{algo.upper()} ({'reach' if r['reached'] else '...'})")
            rx, ry, ryaw = r["xs"][kk], r["ys"][kk], r["yaws"][kk]
            ax.add_patch(plt.Circle((rx, ry), r["rob_r"], color=col, zorder=7))
            ax.plot([rx, rx + math.cos(ryaw) * r["rob_r"] * 1.8],
                    [ry, ry + math.sin(ryaw) * r["rob_r"] * 1.8], color="white", lw=1.5, zorder=8)
        ax.legend(fontsize=7, loc="upper center", ncol=3)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        frames.append(Image.fromarray(buf.reshape(h, w, 4)[:, :, :3].copy()))
        plt.close(fig)

    if frames:
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        dur = max(int(round(1000.0 / float(fps))), 20)
        pal = [im.convert("P", palette=Image.ADAPTIVE, colors=64) for im in frames]
        pal[0].save(filename, format="GIF", save_all=True, append_images=pal[1:],
                    duration=dur, loop=0, optimize=True, disposal=2)
    return {"gif": filename if frames else None, "locker_id": int(locker_id),
            "frames": len(frames), "per_model": summary}
