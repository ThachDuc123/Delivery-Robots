"""BƯỚC 2 — Train AI Robust (chống overfit) trên 1000 bản đồ đã sinh.

Dùng ĐÚNG cách train ms_mixed cũ: RecurrentPPO (MlpLstmPolicy, 32-chiều obs),
VecNormalize, n_steps=1024, lr-schedule, ent_coef cao để thăm dò. Khác biệt:
  * mỗi episode lấy NGẪU NHIÊN 1 trong 1000 map ở data/maps/ (domain randomization
    mạnh) -> policy buộc phải tổng quát hoá, không overfit 1 toà nhà.
  * reward nâng cấp (đã thêm trong multistop_env): +bám waypoint / +giữ k/c tường /
    +đi thẳng ổn định ; -va chạm / -sát tường / -zigzag / -angular velocity lớn /
    -đổi lái liên tục.
  * Smart Early-Stopping: EvalCallback + StopTrainingOnNoModelImprovement -> tự
    dừng khi reward eval hết cải thiện (chống train mù / overfit).

Run: .venv\\Scripts\\python.exe train_ms_mixed_robust.py --timesteps 8000000 --n-envs 8 --subproc
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

MAP_DIR = os.path.join(_HERE, "data", "maps")
CFG = dict(n_lidar=24, lidar_range=5.0, lookahead=1.6, max_steps=2000,
           grace_steps=18, collision_grace=25, reverse_frac=0.4,
           map_dir=MAP_DIR,            # <-- sample one of the 1000 saved maps / episode
           domain_random=True,         # sensor noise + actuation jitter
           # --- CÁCH 1: chuẩn hoá + scale lại reward (trị explained-variance thấp) ---
           # giảm spike thưởng cuối (dock/stop/jam/collide) để chúng không lấn át;
           # tăng progress dense; clip phần shaping -> return đồng đều giữa 1000 map
           # -> Critic fit tốt hơn -> explained_variance & độ ổn định tăng.
           w_progress=2.5,             # 1.5 -> 2.5 (dense quan trọng hơn)
           w_dock=30.0,                # 100 -> 30
           w_stop=12.0,                # 50 -> 12
           w_jam=30.0,                 # 60 -> 30
           w_collide=4.0,              # 8 -> 4
           shaping_clip=3.0)           # chặn outlier penalty cộng dồn ở ±3


def make_env(rank, seed):
    def _i():
        e = MultiStopEnv(config=CFG); e.reset(seed=seed + rank); return e
    return _i


def lr_schedule(initial=3e-4, final=1e-4):
    def f(progress_remaining):
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
    ap.add_argument("--timesteps", type=int, default=8_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=os.path.join(_HERE, "runs", "ms_mixed_robust"))
    ap.add_argument("--subproc", action="store_true")
    ap.add_argument("--no-early-stop", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(os.path.join(MAP_DIR, "index.json")):
        sys.exit("Chưa có data/maps/index.json — chạy generate_training_maps.py trước (BƯỚC 1).")

    fns = [make_env(i, args.seed) for i in range(args.n_envs)]
    venv = SubprocVecEnv(fns) if (args.subproc and args.n_envs > 1) else DummyVecEnv(fns)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = RecurrentPPO(
        "MlpLstmPolicy", venv, verbose=1, seed=args.seed,
        tensorboard_log=os.path.join(_HERE, "runs", "tb"),
        n_steps=1024, batch_size=512, n_epochs=10, gamma=0.997, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.015, learning_rate=lr_schedule(), max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256], lstm_hidden_size=128,
                           n_lstm_layers=1, enable_critic_lstm=True),
    )
    callbacks = [Stats(),
                 CheckpointCallback(save_freq=max(args.timesteps // (args.n_envs * 8), 1),
                                    save_path=os.path.join(_HERE, "runs", "ckpt_robust"),
                                    name_prefix="ms_robust")]
    if not args.no_early_stop:
        from stable_baselines3.common.callbacks import (EvalCallback,
                                                        StopTrainingOnNoModelImprovement)
        eval_env = VecNormalize(DummyVecEnv([make_env(99, args.seed)]),
                                norm_obs=True, norm_reward=False, training=False)
        eval_env.obs_rms = venv.obs_rms
        stop_cb = StopTrainingOnNoModelImprovement(max_no_improvement_evals=6,
                                                   min_evals=10, verbose=1)
        callbacks.append(EvalCallback(eval_env, eval_freq=max(40000 // args.n_envs, 1),
                                      n_eval_episodes=12, deterministic=True,
                                      callback_after_eval=stop_cb, verbose=1,
                                      best_model_save_path=os.path.join(_HERE, "runs", "best_robust")))
    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=False)
    model.save(args.save); venv.save(args.save + "_vecnorm.pkl")
    print(f"saved -> {args.save}.zip + vecnorm")


if __name__ == "__main__":
    main()
