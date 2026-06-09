"""Rollout visualization: record a trained policy as an annotated GIF.

Uses the env's synthetic overhead camera (``render_mode="rgb_array"``) -- for
HUMAN VIEWING ONLY; it is never part of the observation (the policy stays
strictly sensor-only). Each frame is annotated (with Pillow) to show the active
safety-shield state (GO / SLOW / WAIT / SIDESTEP / BEEP) and a DELIVERED banner.
The red dot hovering over a locker (drawn in the 3D scene by the env) marks the
locker the robot is currently routing to.
"""

from __future__ import annotations

import copy
import os
from typing import Dict, List, Optional

import numpy as np

from delivery_rl.configs.loader import default_config_path, load_config
from delivery_rl.envs.corridor_delivery_env import CorridorDeliveryEnv

ALGO_FILES = {
    "ppo": ("ppo", "ppo_final.zip"),
    "sac": ("sac", "sac_final.zip"),
    "td3": ("td3", "td3_final.zip"),
}

_STATUS_TEXT = {
    "clear": "GO",
    "slow_follow": "SLOW (follow)",
    "yield_wait": "WAIT (person)",
    "sidestep": "SIDESTEP",
    "blocked_beep": "BEEP! blocked",
}


def list_trained_models(repo_root: str) -> Dict[str, str]:
    """Return {algo: path} for every runs/<algo>/<algo>_final.zip that exists."""
    runs = os.path.join(repo_root, "delivery_rl", "runs")
    found = {}
    for algo, (sub, fname) in ALGO_FILES.items():
        path = os.path.join(runs, sub, fname)
        if os.path.isfile(path):
            found[algo] = path
    return found


def _annotate(frame: np.ndarray, meta: dict, label: str) -> np.ndarray:
    from PIL import Image, ImageDraw
    img = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
    d = ImageDraw.Draw(img)
    status = _STATUS_TEXT.get(meta["status"], meta["status"])
    d.rectangle([0, 0, img.width, 16], fill=(0, 0, 0))
    d.text((4, 3), f"{label} | {status}", fill=(255, 255, 255))
    if meta["n"] > 0 and meta["deliv"] >= meta["n"]:
        d.text((img.width - 78, 3), "DELIVERED", fill=(90, 255, 130))
    if meta["beep"]:
        d.rectangle([0, 18, 66, 36], fill=(220, 30, 30))
        d.text((6, 22), "BEEP!", fill=(255, 255, 255))
    return np.asarray(img, dtype=np.uint8)


def _save_gif(path: str, frames: List[np.ndarray], fps: int) -> None:
    """Write an animated GIF. Pillow with explicit palette conversion is the
    primary writer (some imageio GIF backends silently emit a 0-byte file here).
    Any failure is recorded to ``<path>.err`` and re-raised so it is never
    swallowed into a silent 0-byte result."""
    duration_ms = max(int(round(1000.0 / float(fps))), 20)
    errfile = path + ".err"
    if os.path.exists(errfile):
        os.remove(errfile)
    try:
        from PIL import Image
        # convert each RGB frame to a paletted ('P') image -- the native GIF mode.
        # optimize=True keeps the file small so the notebook renderer (e.g. the
        # VSCode webview) is not overwhelmed by many looping animations.
        imgs = [Image.fromarray(np.ascontiguousarray(f, dtype=np.uint8)).convert(
            "P", palette=Image.ADAPTIVE, colors=64) for f in frames]
        imgs[0].save(path, format="GIF", save_all=True, append_images=imgs[1:],
                     duration=duration_ms, loop=0, optimize=True, disposal=2)
        if os.path.getsize(path) > 0:
            return
        raise RuntimeError("Pillow wrote a 0-byte GIF")
    except Exception as exc:  # noqa: BLE001 -- record and re-raise
        import traceback
        with open(errfile, "w", encoding="utf-8") as fh:
            fh.write(f"frames={len(frames)} shape="
                     f"{None if not frames else np.asarray(frames[0]).shape}\n")
            traceback.print_exc(file=fh)
        raise


def record_rollout_gif(
    model,
    level: int = 0,
    filename: str = "rollout.gif",
    max_steps: int = 900,
    fps: int = 20,
    seed: int = 2024,
    max_frames: int = 200,
    stop_after_deliver: bool = True,
    tail_steps: int = 24,
    scenario_override: Optional[dict] = None,
    label: str = "",
    config: Optional[dict] = None,
) -> Dict:
    """Run one deterministic episode and write an annotated GIF.

    The episode runs until the robot reaches and delivers to its target (plus a
    short tail), or until termination/``max_steps`` -- so the GIF shows the whole
    trip to the destination rather than cutting off early.
    """
    cfg = copy.deepcopy(config) if config is not None else load_config(default_config_path())
    cfg["env"]["curriculum"]["level"] = level
    cfg["env"]["max_episode_steps"] = max_steps
    if scenario_override:
        cfg["env"]["scenario_override"] = scenario_override

    env = CorridorDeliveryEnv(config=cfg, render_mode="rgb_array")
    obs, info = env.reset(seed=seed)
    n_parcels = info["num_parcels"]
    records = []  # (frame, meta)

    def snap(meta):
        f = env.render()
        if f is not None:
            records.append((np.asarray(f, dtype=np.uint8), meta))

    snap({"status": "clear", "beep": False, "deliv": 0, "n": n_parcels})
    done, steps, deliver_step = False, 0, None
    last_info = info
    while not done and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, term, trunc, info = env.step(action)
        last_info = info
        steps += 1
        snap({"status": info.get("shield_status", "clear"), "beep": info.get("beep", False),
              "deliv": info.get("deliveries_done", 0), "n": n_parcels})
        done = term or trunc
        if stop_after_deliver and deliver_step is None and info.get("deliveries_done", 0) >= n_parcels:
            deliver_step = steps
        if deliver_step is not None and steps >= deliver_step + tail_steps:
            break
    env.close()

    # subsample evenly to keep the GIF light, but cover the whole trip
    if len(records) > max_frames:
        idx = np.linspace(0, len(records) - 1, max_frames).astype(int)
        records = [records[i] for i in idx]
    frames = [_annotate(f, meta, label) for f, meta in records]

    if frames:
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        _save_gif(filename, frames, fps)
    out = dict(last_info)
    out["steps"] = steps
    out["frames"] = len(frames)
    out["gif"] = filename if frames else None
    return out


def default_scenarios() -> List[dict]:
    """Visual scenarios for the GIF gallery.

    All use curriculum **level 0** (nearest locker -- the task the policies were
    trained on, so the robot can actually reach the goal) and switch on one
    capability via the override: plain navigation, static obstacles, or moving
    pedestrians (handled by the reactive safety shield)."""
    return [
        {"name": "clear", "level": 0, "override": {}},
        {"name": "obstacles", "level": 0, "override": {"num_obstacles": 4, "num_pedestrians": 0}},
        {"name": "pedestrians", "level": 0, "override": {"num_obstacles": 0, "num_pedestrians": 3}},
    ]


def reaches_goal(model, level, override, seed, max_steps=900) -> tuple:
    """Headless pre-check: does the robot deliver within max_steps for this seed?
    Returns (reached, steps). Used to pick a good seed for the GIF."""
    cfg = load_config(default_config_path())
    cfg["env"]["curriculum"]["level"] = level
    cfg["env"]["max_episode_steps"] = max_steps
    if override:
        cfg["env"]["scenario_override"] = override
    env = CorridorDeliveryEnv(config=cfg)
    obs, info = env.reset(seed=seed)
    n = info["num_parcels"]
    done, steps = False, 0
    while not done and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, term, trunc, info = env.step(action)
        steps += 1
        if info["deliveries_done"] >= n:
            env.close()
            return True, steps
        done = term or trunc
    env.close()
    return False, steps


def pick_seed(model, level, override, seeds) -> int:
    """Return the first seed for which the robot reaches the goal (else seeds[0])."""
    for sd in seeds:
        reached, _ = reaches_goal(model, level, override, sd)
        if reached:
            return sd
    return seeds[0]
