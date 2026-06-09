"""Train PPO to follow planned routes on the fixed delivery maps (stable driving).

Same recipe that worked for nav2d (frame-stack + parallel + VecNormalize) but on
DeliveryFollowEnv, whose reward is shaped for steady, on-the-line driving. The
resulting policy is what hybrid_runner uses as the local controller, replacing
the wide-corridor policy that weaved on these tighter maps.

Run: python nav2d/train_delivery.py --timesteps 1200000 --n-envs 8 --subproc
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecFrameStack, VecNormalize)
from delivery_train_env import DeliveryFollowEnv

CFG = dict(n_lidar=24, lidar_range=5.0, max_steps=700, lookahead=1.6)


def make_env(rank, seed):
    def _i():
        e = DeliveryFollowEnv(config=CFG); e.reset(seed=seed + rank); return e
    return _i


class Stats(BaseCallback):
    def __init__(self): super().__init__(); self.ar=[]; self.co=[]
    def _on_step(self):
        for inf, d in zip(self.locals["infos"], self.locals["dones"]):
            if d:
                self.ar.append(int(inf.get("arrived", False)))
                self.co.append(int(inf.get("collision", False)))
                if len(self.ar) % 50 == 0:
                    self.logger.record("deliv/arrive_rate", float(np.mean(self.ar[-200:])))
                    self.logger.record("deliv/collision_rate", float(np.mean(self.co[-200:])))
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=1_200_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=os.path.join(_HERE, "runs", "ppo_delivery"))
    ap.add_argument("--subproc", action="store_true")
    ap.add_argument("--warm", default="", help="warm-start from this model .zip (+ _vecnorm.pkl)")
    args = ap.parse_args()

    fns = [make_env(i, args.seed) for i in range(args.n_envs)]
    venv = SubprocVecEnv(fns) if (args.subproc and args.n_envs > 1) else DummyVecEnv(fns)
    venv = VecFrameStack(venv, 4)
    if args.warm and os.path.isfile(args.warm + "_vecnorm.pkl"):
        venv = VecNormalize.load(args.warm + "_vecnorm.pkl", venv)
        venv.training = True; venv.norm_reward = True
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    if args.warm and os.path.isfile(args.warm + ".zip"):
        model = PPO.load(args.warm, env=venv, tensorboard_log=os.path.join(_HERE, "runs", "tb"))
        print(f"warm-started from {args.warm}.zip")
    else:
        model = PPO("MlpPolicy", venv, verbose=1, seed=args.seed,
                    tensorboard_log=os.path.join(_HERE, "runs", "tb"),
                    n_steps=1024, batch_size=512, n_epochs=10, gamma=0.997, gae_lambda=0.95,
                    clip_range=0.2, ent_coef=0.004, learning_rate=3e-4, max_grad_norm=0.5,
                    policy_kwargs=dict(net_arch=[256, 256]))
    ckpt = CheckpointCallback(save_freq=max(args.timesteps // (args.n_envs * 6), 1),
                              save_path=os.path.join(_HERE, "runs", "ckpt_deliv"),
                              name_prefix="ppo_delivery")
    model.learn(total_timesteps=args.timesteps, callback=[Stats(), ckpt], progress_bar=False)
    model.save(args.save); venv.save(args.save + "_vecnorm.pkl")
    print(f"saved -> {args.save}.zip + vecnorm")


if __name__ == "__main__":
    main()
