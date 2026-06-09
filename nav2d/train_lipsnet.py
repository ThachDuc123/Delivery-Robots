"""Thử nghiệm LipsNet-style: khống chế Lipschitz constant của ACTOR (gốc của
'action fluctuation' mà reward không hạ nổi) bằng SPECTRAL NORMALIZATION trên các
lớp Linear của mạng actor. Đây là bản stand-in thực dụng cho LipsNet (ICML 2023):
cùng tinh thần "bound Lipschitz của actor -> action mượt ở tầng mạng", chạy được
ngay không cần module MGN gốc.

Train MỚI (không warm-start được vì đổi parametrization mạng) trên track_env.
Giữ NGUYÊN mọi model cũ. Lưu -> ms_guided_lipsnet.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch as th

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from track_env import TrackEnv

CFG = dict(domain_random=True, K_guide=1.0, max_steps=1500, w_dw=0.3, w_omega=0.02)


def make_env(rank, seed):
    def _i():
        e = TrackEnv(config=CFG); e.reset(seed=seed + rank); return e
    return _i


def lr_schedule(initial=3e-4, final=1e-4):
    def f(p): return final + (initial - final) * p
    return f


def constrain_actor_lipschitz(policy):
    """Áp spectral_norm lên các Linear của ACTOR (mlp_extractor.policy_net +
    action_net) -> bound Lipschitz -> action mượt. KHÔNG đụng critic."""
    from torch.nn.utils.parametrizations import spectral_norm
    n = 0
    targets = []
    if hasattr(policy.mlp_extractor, "policy_net"):
        targets.append(policy.mlp_extractor.policy_net)
    if hasattr(policy, "action_net"):
        targets.append(policy.action_net)
    for mod in targets:
        if isinstance(mod, th.nn.Linear):
            spectral_norm(mod); n += 1
        else:
            for layer in mod.modules():
                if isinstance(layer, th.nn.Linear):
                    spectral_norm(layer); n += 1
    return n


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
    ap.add_argument("--timesteps", type=int, default=3_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=os.path.join(_HERE, "runs", "ms_guided_lipsnet"))
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
    n = constrain_actor_lipschitz(model.policy)
    # rebuild optimizer để bám đúng tham số sau spectral_norm
    model.policy.optimizer = model.policy.optimizer_class(
        model.policy.parameters(), lr=lr_schedule()(1.0),
        **(model.policy.optimizer_kwargs or {}))
    print(f"Đã áp spectral_norm lên {n} lớp Linear của actor (Lipschitz-constrained).")

    cbs = [Stats()]
    if not args.no_early_stop:
        from stable_baselines3.common.callbacks import (EvalCallback,
                                                        StopTrainingOnNoModelImprovement)
        eval_env = VecNormalize(DummyVecEnv([make_env(99, args.seed)]),
                                norm_obs=True, norm_reward=False, training=False)
        eval_env.obs_rms = venv.obs_rms
        stop_cb = StopTrainingOnNoModelImprovement(max_no_improvement_evals=6, min_evals=10, verbose=1)
        cbs.append(EvalCallback(eval_env, eval_freq=max(40000 // args.n_envs, 1),
                                n_eval_episodes=12, deterministic=True,
                                callback_after_eval=stop_cb, verbose=1,
                                best_model_save_path=os.path.join(_HERE, "runs", "best_lipsnet")))
    model.learn(total_timesteps=args.timesteps, callback=cbs, progress_bar=False)
    model.save(args.save); venv.save(args.save + "_vecnorm.pkl")
    print(f"saved -> {args.save}.zip + vecnorm")


if __name__ == "__main__":
    main()
