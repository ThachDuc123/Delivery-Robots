"""Evaluate the trained PPO nav2d policy on many UNSEEN maps + make GIFs.

Loads the saved VecNormalize stats so observations are normalised exactly as in
training, then runs deterministic rollouts across all corridor styles (seeds the
policy never trained on) and reports reach-goal / round-trip / collision rates,
plus a path grid and a few GIFs.
"""

from __future__ import annotations

import argparse
import os
import sys
import collections

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

from nav_env import Nav2DEnv
from world2d import STYLES
from render2d import record_gif, plot_paths

ENV_CONFIG = dict(n_lidar=24, lidar_range=5.0, max_steps=800, round_trip=True)
FRAME_STACK = 4


def load(model_path, vec_path):
    model = PPO.load(model_path)
    # rebuild a vecnormalize to recover obs mean/var for manual normalisation
    venv = VecFrameStack(DummyVecEnv([lambda: Nav2DEnv(config=ENV_CONFIG)]), FRAME_STACK)
    venv = VecNormalize.load(vec_path, venv)
    venv.training = False
    venv.norm_reward = False
    return model, venv


def _normalizer(venv):
    mean = venv.obs_rms.mean.astype(np.float32)
    var = venv.obs_rms.var.astype(np.float32)
    eps = venv.epsilon
    clip = venv.clip_obs

    def norm(stacked):
        return np.clip((stacked - mean) / np.sqrt(var + eps), -clip, clip).astype(np.float32)
    return norm


def run_episode(model, raw_env, norm, seed, style, max_steps=800):
    o, info = raw_env.reset(seed=seed, options={"style": style})
    d = o.shape[0]
    stack = np.tile(o, FRAME_STACK)
    done = False; steps = 0; rg = rt = coll = False
    while not done and steps < max_steps:
        a, _ = model.predict(norm(stack), deterministic=True)
        o, r, term, trunc, info = raw_env.step(a)
        stack = np.concatenate([stack[d:], o])
        steps += 1
        rg = rg or info["reached_goal"]; rt = rt or info["round_trip"]
        coll = coll or info["collision"]
        done = term or trunc
    return {"reached_goal": rg, "round_trip": rt, "collision": coll, "steps": steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(_HERE, "runs", "ppo_nav2d"))
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--gifs", action="store_true")
    args = ap.parse_args()

    model, venv = load(args.model, args.model + "_vecnorm.pkl")
    norm = _normalizer(venv)
    raw = Nav2DEnv(config=ENV_CONFIG)

    print(f"=== Eval PPO nav2d ({args.episodes} unseen maps / style) ===")
    overall = collections.Counter()
    rows = []
    for style in STYLES:
        rg = rt = coll = stp = 0
        for sd in range(args.episodes):
            res = run_episode(model, raw, norm, seed=9000 + sd, style=style)
            rg += res["reached_goal"]; rt += res["round_trip"]
            coll += res["collision"]; stp += res["steps"]
        n = args.episodes
        rows.append((style, rg / n, rt / n, coll / n, stp / n))
        overall["rg"] += rg; overall["rt"] += rt; overall["coll"] += coll; overall["n"] += n
        print(f"  {style:9s} reach {rg/n*100:5.0f}%  round-trip {rt/n*100:5.0f}%  "
              f"collision {coll/n*100:5.0f}%  avg_steps {stp/n:5.0f}")
    N = overall["n"]
    print(f"  {'ALL':9s} reach {overall['rg']/N*100:5.0f}%  round-trip {overall['rt']/N*100:5.0f}%  "
          f"collision {overall['coll']/N*100:5.0f}%")

    # path grid (one per style) -- plot_paths resets the env per style, so it
    # needs a wrapper whose frame-stack also resets each style; pass a factory.
    figpath = os.path.join(_HERE, "results", "paths_grid.png")
    plot_paths(WrapFactory(model, norm), raw, list(STYLES),
               seeds=[7000 + i for i in range(len(STYLES))], savepath=figpath)
    print("saved", figpath)

    if args.gifs:
        gdir = os.path.join(_HERE, "results", "gifs")
        for style in STYLES:
            out = record_gif(model_wrap(model, norm), raw,
                             os.path.join(gdir, f"nav2d_{style}.gif"),
                             seed=7700, style=style, max_steps=800, fps=18, max_frames=110)
            print(f"  gif {style:9s} round_trip={out['round_trip']} reached={out['reached_goal']}")


class model_wrap:
    """Wrap (model + obs normalizer + frame stack) so render2d can call .predict(obs).

    render2d/record_gif call env.reset() then predict() each step. We detect a new
    episode by the prev_action slot being ~0 right after reset (phase/prev_action
    cleared), and also expose reset(); but to be robust we re-tile whenever the
    caller signals a fresh start via the `new_episode` flag on the wrapper."""
    def __init__(self, model, norm):
        self.model = model; self.norm = norm; self._stack = None; self._d = None
        self._first = True
    def predict(self, obs, deterministic=True):
        o = np.asarray(obs, dtype=np.float32)
        if self._stack is None or self._d != o.shape[0] or self._first:
            self._d = o.shape[0]; self._stack = np.tile(o, FRAME_STACK); self._first = False
        else:
            self._stack = np.concatenate([self._stack[self._d:], o])
        a, _ = self.model.predict(self.norm(self._stack), deterministic=deterministic)
        return a, None


class WrapFactory:
    """predict() that auto-resets its frame stack when plot_paths starts a new
    style. We detect a fresh episode by env trail length via a step counter that
    plot_paths implicitly restarts (each style calls reset -> first predict)."""
    def __init__(self, model, norm):
        self.model = model; self.norm = norm
        self._stack = None; self._d = None; self._last_phase_seen = None
        self._steps = 0
    def predict(self, obs, deterministic=True):
        o = np.asarray(obs, dtype=np.float32)
        # phase flag is the LAST obs element; on reset phase=0 and prev_action=0.
        prev_act = o[-3:-1]
        fresh = (self._stack is None or self._d != o.shape[0]
                 or (abs(prev_act[0]) < 1e-6 and abs(prev_act[1]) < 1e-6 and self._steps > 3))
        if fresh:
            self._d = o.shape[0]; self._stack = np.tile(o, FRAME_STACK); self._steps = 0
        else:
            self._stack = np.concatenate([self._stack[self._d:], o])
        self._steps += 1
        a, _ = self.model.predict(self.norm(self._stack), deterministic=deterministic)
        return a, None


if __name__ == "__main__":
    main()
