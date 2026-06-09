"""Train PPO on the 2D sensor-only corridor navigation env.

Improvements layered on plain PPO so the robot navigates robustly and far:
  * **VecFrameStack(4)**: stacks the last 4 observations so the policy has short-
    term memory (escape dead-ends, take arcs smoothly, no left/right dithering).
  * **SubprocVecEnv (parallel envs)**: many corridors generated at once -> fast,
    and the policy sees a huge variety of layouts each update -> generalises.
  * **VecNormalize**: normalises observations + returns for stable learning.
  * **Curriculum by style mix**: easy styles early, all styles later (optional).

Run:
    python nav2d/train_ppo.py --timesteps 1500000 --n-envs 8
    tensorboard --logdir nav2d/runs
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecFrameStack, VecNormalize)

from nav_env import Nav2DEnv

ENV_CONFIG = dict(n_lidar=24, lidar_range=5.0, max_steps=1000, round_trip=True,
                  lidar_noise=0.0, world_kind=os.environ.get("NAV2D_WORLD_KIND", "simple"),
                  # branch/tight-corridor tuning (clearance + light recoverable bumps)
                  w_clear=0.06, clear_thresh=0.18, w_bump=0.3, collision_grace=25)


def make_env(rank: int, seed: int, config: dict):
    def _init():
        env = Nav2DEnv(config=config)
        env.reset(seed=seed + rank)
        return env
    return _init


class StatsCallback(BaseCallback):
    """Logs reach/round-trip rates to TensorBoard from the info dicts."""
    def __init__(self):
        super().__init__()
        self.reach = []; self.rt = []; self.coll = []

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                self.reach.append(int(info.get("reached_goal", False)))
                self.rt.append(int(info.get("round_trip", False)))
                self.coll.append(int(info.get("collision", False)))
                if len(self.reach) % 50 == 0:
                    w = self.reach[-200:]; r = self.rt[-200:]; c = self.coll[-200:]
                    self.logger.record("nav/reach_goal_rate", float(np.mean(w)))
                    self.logger.record("nav/round_trip_rate", float(np.mean(r)))
                    self.logger.record("nav/collision_rate", float(np.mean(c)))
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=1_500_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frame-stack", type=int, default=4)
    ap.add_argument("--save", type=str, default=os.path.join(_HERE, "runs", "ppo_nav2d"))
    ap.add_argument("--subproc", action="store_true", help="use SubprocVecEnv (faster)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    tb = os.path.join(_HERE, "runs", "tb")

    fns = [make_env(i, args.seed, ENV_CONFIG) for i in range(args.n_envs)]
    venv = SubprocVecEnv(fns) if (args.subproc and args.n_envs > 1) else DummyVecEnv(fns)
    venv = VecFrameStack(venv, n_stack=args.frame_stack)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy", venv, verbose=1, seed=args.seed, tensorboard_log=tb,
        n_steps=1024, batch_size=512, n_epochs=10, gamma=0.995, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.004, learning_rate=3e-4, max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
    )
    ckpt = CheckpointCallback(save_freq=max(args.timesteps // (args.n_envs * 6), 1),
                              save_path=os.path.join(_HERE, "runs", "ckpt"),
                              name_prefix="ppo_nav2d")
    model.learn(total_timesteps=args.timesteps, callback=[StatsCallback(), ckpt],
                progress_bar=False)
    model.save(args.save)
    venv.save(args.save + "_vecnorm.pkl")
    print(f"saved model -> {args.save}.zip  + vecnorm")


if __name__ == "__main__":
    main()
