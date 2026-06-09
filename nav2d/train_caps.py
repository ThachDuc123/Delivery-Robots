"""Update model cũ (ms_guided_smooth) + CAPS spatial smoothness.
Warm-start: nạp trọng số ms_guided_smooth -> train tiếp với CapsPPO -> ms_guided_caps.
Giữ NGUYÊN mọi model cũ.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from track_env import TrackEnv
from caps_ppo import CapsPPO

# CÙNG cấu hình env đã tạo ra ms_guided_smooth (K=1.0, w_dw=0.3) để chỉ THÊM CAPS spatial
CFG = dict(domain_random=True, K_guide=1.0, max_steps=1500, w_dw=0.3, w_omega=0.02)
BASE = "ms_guided_smooth"


def make_env(rank, seed):
    def _i():
        e = TrackEnv(config=CFG); e.reset(seed=seed + rank); return e
    return _i


def lr_schedule(initial=2e-4, final=5e-5):   # lr nhỏ hơn vì warm-start (fine-tune)
    def f(p): return final + (initial - final) * p
    return f


class Stats(BaseCallback):
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
    ap.add_argument("--timesteps", type=int, default=2_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=os.path.join(_HERE, "runs", "ms_guided_caps"))
    ap.add_argument("--subproc", action="store_true")
    ap.add_argument("--caps-lambda", type=float, default=0.05)
    ap.add_argument("--caps-sigma", type=float, default=0.05)
    ap.add_argument("--no-early-stop", action="store_true")
    args = ap.parse_args()

    fns = [make_env(i, args.seed) for i in range(args.n_envs)]
    venv = SubprocVecEnv(fns) if (args.subproc and args.n_envs > 1) else DummyVecEnv(fns)
    # warm-start cả VecNormalize stats từ model cũ
    vn = os.path.join(_HERE, "runs", f"{BASE}_vecnorm.pkl")
    if os.path.isfile(vn):
        venv = VecNormalize.load(vn, venv); venv.training = True; venv.norm_reward = True
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = CapsPPO(
        "MlpPolicy", venv, verbose=1, seed=args.seed,
        tensorboard_log=os.path.join(_HERE, "runs", "tb"),
        n_steps=1024, batch_size=512, n_epochs=10, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.005, learning_rate=lr_schedule(), max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
        caps_lambda=args.caps_lambda, caps_sigma=args.caps_sigma,
    )
    # nạp TRỌNG SỐ model cũ (warm-start / update)
    base_zip = os.path.join(_HERE, "runs", f"{BASE}.zip")
    if os.path.isfile(base_zip):
        old = PPO.load(base_zip)
        model.policy.load_state_dict(old.policy.state_dict())
        print(f"warm-started policy from {BASE}.zip + CAPS (lambda={args.caps_lambda}, sigma={args.caps_sigma})")
    else:
        print(f"WARNING: {BASE}.zip không thấy -> train CAPS từ đầu")

    cbs = [Stats()]
    if not args.no_early_stop:
        from stable_baselines3.common.callbacks import (EvalCallback,
                                                        StopTrainingOnNoModelImprovement)
        eval_env = VecNormalize(DummyVecEnv([make_env(99, args.seed)]),
                                norm_obs=True, norm_reward=False, training=False)
        eval_env.obs_rms = venv.obs_rms
        stop_cb = StopTrainingOnNoModelImprovement(max_no_improvement_evals=6, min_evals=8, verbose=1)
        cbs.append(EvalCallback(eval_env, eval_freq=max(40000 // args.n_envs, 1),
                                n_eval_episodes=12, deterministic=True,
                                callback_after_eval=stop_cb, verbose=1,
                                best_model_save_path=os.path.join(_HERE, "runs", "best_caps")))
    model.learn(total_timesteps=args.timesteps, callback=cbs, progress_bar=False)
    model.save(args.save); venv.save(args.save + "_vecnorm.pkl")
    print(f"saved -> {args.save}.zip + vecnorm")


if __name__ == "__main__":
    main()
