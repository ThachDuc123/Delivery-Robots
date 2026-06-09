"""Train Guided-RL (PPO) trên track_env: học LÁI như Pure-Pursuit nhưng giữ tính
cách RL (dùng cảm biến để né, không học vẹt map).

- State ego-relative (track_env) -> zero-shot, không nhớ map.
- Reward = R_nav + R_collision - K*(omega_RL - omega_PP)^2 (Pure-Pursuit chấm điểm).
- MlpPolicy (state đã Markov cho path-tracking -> không cần LSTM).
- Early-Stopping chống overfit. Lưu model MỚI `ms_guided`, GIỮ NGUYÊN model cũ.

Run: .venv\\Scripts\\python.exe train_guided_rl.py --timesteps 4000000 --n-envs 8 --subproc
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from track_env import TrackEnv

CFG = dict(domain_random=True, K_guide=1.2, max_steps=1500,
           w_dw=0.6, w_omega=0.04)   # ĐỘ MƯỢT TỐI ĐA: phạt Δω mạnh gấp đôi (mục tiêu <= PP)


def make_env(rank, seed):
    def _i():
        e = TrackEnv(config=CFG); e.reset(seed=seed + rank); return e
    return _i


def lr_schedule(initial=3e-4, final=1e-4):
    def f(p): return final + (initial - final) * p
    return f


class Stats(BaseCallback):
    """Log: arrive rate, độ bám tim đường (|e_y|), khoảng cách lái so với Pure-Pursuit."""
    def __init__(self): super().__init__(); self.arr=[]; self.gap=[]; self.col=[]
    def _on_step(self):
        for inf, d in zip(self.locals["infos"], self.locals["dones"]):
            if "omega_pp" in inf:
                self.gap.append(abs(inf["omega_rl"] - inf["omega_pp"]))
            if d:
                self.arr.append(int(inf.get("arrived", False)))
                self.col.append(int(inf.get("collision", False)))
                if len(self.arr) % 50 == 0:
                    self.logger.record("guide/arrive_rate", float(np.mean(self.arr[-200:])))
                    self.logger.record("guide/collision_rate", float(np.mean(self.col[-200:])))
                    self.logger.record("guide/omega_gap_vs_PP", float(np.mean(self.gap[-4000:])))
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=4_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=os.path.join(_HERE, "runs", "ms_guided"))
    ap.add_argument("--subproc", action="store_true")
    ap.add_argument("--no-early-stop", action="store_true")
    args = ap.parse_args()

    fns = [make_env(i, args.seed) for i in range(args.n_envs)]
    venv = SubprocVecEnv(fns) if (args.subproc and args.n_envs > 1) else DummyVecEnv(fns)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy", venv, verbose=1, seed=args.seed,
        tensorboard_log=os.path.join(_HERE, "runs", "tb"),
        n_steps=1024, batch_size=512, n_epochs=10, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.005, learning_rate=lr_schedule(), max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
    )
    callbacks = [Stats(),
                 CheckpointCallback(save_freq=max(args.timesteps // (args.n_envs * 8), 1),
                                    save_path=os.path.join(_HERE, "runs", "ckpt_guided"),
                                    name_prefix="ms_guided")]
    if not args.no_early_stop:
        from stable_baselines3.common.callbacks import (EvalCallback,
                                                        StopTrainingOnNoModelImprovement)
        eval_env = VecNormalize(DummyVecEnv([make_env(99, args.seed)]),
                                norm_obs=True, norm_reward=False, training=False)
        eval_env.obs_rms = venv.obs_rms
        stop_cb = StopTrainingOnNoModelImprovement(max_no_improvement_evals=6, min_evals=10, verbose=1)
        callbacks.append(EvalCallback(eval_env, eval_freq=max(40000 // args.n_envs, 1),
                                      n_eval_episodes=12, deterministic=True,
                                      callback_after_eval=stop_cb, verbose=1,
                                      best_model_save_path=os.path.join(_HERE, "runs", "best_guided")))
    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=False)
    model.save(args.save); venv.save(args.save + "_vecnorm.pkl")
    print(f"saved -> {args.save}.zip + vecnorm")


if __name__ == "__main__":
    main()
