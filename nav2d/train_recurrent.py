"""Train RecurrentPPO (PPO + LSTM) on the 2D sensor-only navigation env.

Why LSTM here: the weakest case for the feed-forward PPO (even with frame-stack)
was the **U-turn round-trip** -- turning 180 deg in a tight corridor needs memory
of where the robot came from. An LSTM policy carries a hidden state across the
whole episode, so it remembers the corridor it is in and handles long dead-ends /
U-turns better than a fixed 4-frame stack.

Key differences vs train_ppo.py:
  * No VecFrameStack -- the LSTM provides memory itself.
  * MlpLstmPolicy; we keep VecNormalize + parallel envs + TensorBoard.

Run:
    python nav2d/train_recurrent.py --timesteps 1500000 --n-envs 8 --subproc
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv, VecNormalize)

from nav_env import Nav2DEnv

ENV_CONFIG = dict(n_lidar=24, lidar_range=5.0, max_steps=800, round_trip=True,
                  lidar_noise=0.0)


def make_env(rank: int, seed: int, config: dict):
    def _init():
        env = Nav2DEnv(config=config)
        env.reset(seed=seed + rank)
        return env
    return _init


class StatsCallback(BaseCallback):
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
                    self.logger.record("nav/reach_goal_rate", float(np.mean(self.reach[-200:])))
                    self.logger.record("nav/round_trip_rate", float(np.mean(self.rt[-200:])))
                    self.logger.record("nav/collision_rate", float(np.mean(self.coll[-200:])))
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=1_500_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=str, default=os.path.join(_HERE, "runs", "recurrent_nav2d"))
    ap.add_argument("--subproc", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    tb = os.path.join(_HERE, "runs", "tb")

    fns = [make_env(i, args.seed, ENV_CONFIG) for i in range(args.n_envs)]
    venv = SubprocVecEnv(fns) if (args.subproc and args.n_envs > 1) else DummyVecEnv(fns)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = RecurrentPPO(
        "MlpLstmPolicy", venv, verbose=1, seed=args.seed, tensorboard_log=tb,
        n_steps=512, batch_size=256, n_epochs=10, gamma=0.995, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.004, learning_rate=3e-4, max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256], lstm_hidden_size=128,
                           n_lstm_layers=1, enable_critic_lstm=True),
    )
    ckpt = CheckpointCallback(save_freq=max(args.timesteps // (args.n_envs * 6), 1),
                              save_path=os.path.join(_HERE, "runs", "ckpt_rec"),
                              name_prefix="recurrent_nav2d")
    model.learn(total_timesteps=args.timesteps, callback=[StatsCallback(), ckpt],
                progress_bar=False)
    model.save(args.save)
    venv.save(args.save + "_vecnorm.pkl")
    print(f"saved model -> {args.save}.zip + vecnorm")


if __name__ == "__main__":
    main()
