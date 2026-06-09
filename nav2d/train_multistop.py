"""Fresh RecurrentPPO (LSTM) training on the multi-stop delivery env.

LSTM is the deliberate model choice: escaping a dead-end niche and chasing the
flipped lookahead after a stop needs MEMORY ("I came in heading east, so I must
leave heading west"). A feed-forward net only sees the current (or stacked) frame
and gets disoriented at the 180-degree goal transition; the LSTM carries hidden
state across the whole trip.

Fresh train (no warm-start) to avoid the catastrophic-forgetting / stale
VecNormalize that wrecked the previous attempts. Higher entropy keeps the policy
exploring sharp turns / reverse while it's still jammed in a niche, with a
decaying learning-rate schedule for stable convergence.

Run: python nav2d/train_multistop.py --timesteps 2000000 --n-envs 8 --subproc
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from multistop_env import MultiStopEnv

CFG = dict(n_lidar=24, lidar_range=5.0, lookahead=1.6, max_steps=2000,
           grace_steps=18, collision_grace=25, reverse_frac=0.4,
           # Level-2 generalization: a NEW random map every episode + domain
           # randomization (sensor noise + actuation jitter). Set via env var so
           # the same script trains either procedural or fixed-map.
           procedural=os.environ.get("NAV2D_PROCEDURAL", "1") == "1",
           domain_random=os.environ.get("NAV2D_PROCEDURAL", "1") == "1",
           # Stage 3: dynamic pedestrians + LiDAR frame-stacking (NAV2D_PEDS=k).
           n_ped=int(os.environ.get("NAV2D_PEDS", "0")),
           lidar_stack=int(os.environ.get("NAV2D_STACK", "3")),
           # fraction of episodes drawn from the hand-made apartment_complex map
           fixed_mix=float(os.environ.get("NAV2D_FIXEDMIX", "0")))


def make_env(rank, seed):
    def _i():
        e = MultiStopEnv(config=CFG); e.reset(seed=seed + rank); return e
    return _i


def lr_schedule(initial=3e-4, final=1e-4):
    def f(progress_remaining):   # 1 -> 0 over training
        return final + (initial - final) * progress_remaining
    return f


class Stats(BaseCallback):
    def __init__(self): super().__init__(); self.dock=[]; self.stops=[]; self.col=[]
    def _on_step(self):
        for inf, d in zip(self.locals["infos"], self.locals["dones"]):
            if d:
                self.dock.append(int(inf.get("arrived_dock", False)))
                self.stops.append(int(inf.get("stops_done", 0)))
                self.col.append(int(inf.get("collision", False)))
                if len(self.dock) % 50 == 0:
                    self.logger.record("ms/full_trip_rate", float(np.mean(self.dock[-200:])))
                    self.logger.record("ms/avg_stops_done", float(np.mean(self.stops[-200:])))
                    self.logger.record("ms/collision_rate", float(np.mean(self.col[-200:])))
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=2_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=os.path.join(_HERE, "runs", "ms_lstm"))
    ap.add_argument("--subproc", action="store_true")
    ap.add_argument("--warm", default="", help="warm-start from this model (.zip + _vecnorm.pkl)")
    ap.add_argument("--early-stop", action="store_true",
                    help="auto-stop if eval reward plateaus (anti blind-training)")
    args = ap.parse_args()

    fns = [make_env(i, args.seed) for i in range(args.n_envs)]
    venv = SubprocVecEnv(fns) if (args.subproc and args.n_envs > 1) else DummyVecEnv(fns)
    if args.warm and os.path.isfile(args.warm + "_vecnorm.pkl"):
        venv = VecNormalize.load(args.warm + "_vecnorm.pkl", venv)
        venv.training = True; venv.norm_reward = True
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    if args.warm and os.path.isfile(args.warm + ".zip"):
        model = RecurrentPPO.load(args.warm, env=venv,
                                  tensorboard_log=os.path.join(_HERE, "runs", "tb"))
        print(f"warm-started from {args.warm}.zip")
    else:
        model = RecurrentPPO(
            "MlpLstmPolicy", venv, verbose=1, seed=args.seed,
            tensorboard_log=os.path.join(_HERE, "runs", "tb"),
            # n_steps 1024: longer rollouts so the LSTM keeps enough memory to
            # carry through a whole U/S-bend without the sequence being chopped.
            n_steps=1024, batch_size=512, n_epochs=10, gamma=0.997, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.015, learning_rate=lr_schedule(), max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[256, 256], lstm_hidden_size=128,
                               n_lstm_layers=1, enable_critic_lstm=True),
        )
    callbacks = [Stats(),
                 CheckpointCallback(save_freq=max(args.timesteps // (args.n_envs * 8), 1),
                                    save_path=os.path.join(_HERE, "runs", "ckpt_ms"),
                                    name_prefix="ms_lstm")]
    # Smart early-stopping: an eval env + stop-on-no-improvement guards against
    # blind training past the point of diminishing returns / overfitting.
    if args.early_stop:
        from stable_baselines3.common.callbacks import (EvalCallback,
                                                        StopTrainingOnNoModelImprovement)
        eval_env = VecNormalize(DummyVecEnv([make_env(99, args.seed)]),
                                norm_obs=True, norm_reward=False, training=False)
        eval_env.obs_rms = venv.obs_rms
        stop_cb = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5,
                                                   min_evals=8, verbose=1)
        callbacks.append(EvalCallback(eval_env, eval_freq=max(40000 // args.n_envs, 1),
                                      n_eval_episodes=10, deterministic=True,
                                      callback_after_eval=stop_cb, verbose=1,
                                      best_model_save_path=os.path.join(_HERE, "runs", "best_ms")))
    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=False)
    model.save(args.save); venv.save(args.save + "_vecnorm.pkl")
    print(f"saved -> {args.save}.zip + vecnorm")


if __name__ == "__main__":
    main()
