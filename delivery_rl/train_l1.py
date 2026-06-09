"""Train PPO / SAC / TD3 on curriculum L1 (random locker across the whole map),
warm-started from the L0 models so they already know corridor navigation and
only need to learn reaching lockers in rooms / the arc branch.

Writes runs/<algo>/<algo>_final.zip (+ monitor.csv) and results/SUMMARY.md, the
same artifacts the notebook reads -- so after this finishes the 2D map review and
charts show the three models driving to lockers all over the map.

Usage:  python delivery_rl/train_l1.py --timesteps 150000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from delivery_rl.configs.loader import load_config
from delivery_rl.envs.corridor_delivery_env import CorridorDeliveryEnv

ALGOS = {"ppo": PPO, "sac": SAC, "td3": TD3}
_CFG = os.path.join(_HERE, "configs")
_WARM = os.path.join(_HERE, "_L0newmap_backup")


def make_env(config, seed, monitor_path):
    env = Monitor(CorridorDeliveryEnv(config=config), filename=monitor_path)
    env.reset(seed=seed)
    return env


def evaluate(model, config, episodes, seed0, max_steps):
    env = CorridorDeliveryEnv(config=config)
    R, D, C, OK, PL = [], [], [], [], []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed0 + ep)
        n = info["num_parcels"]; done = False; tot = coll = dd = 0
        px, py, _ = env.robot.get_pose(); plen = 0.0
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            tot += r; coll += int(info["collision"]); dd = info["deliveries_done"]
            nx, ny, _ = env.robot.get_pose(); plen += float(np.hypot(nx - px, ny - py)); px, py = nx, ny
            done = term or trunc
        R.append(tot); D.append(dd / max(n, 1)); C.append(coll)
        OK.append(int(info["is_success"])); PL.append(plen)
    env.close()
    return dict(avg_reward=float(np.mean(R)), delivery_rate=float(np.mean(D)),
                success_rate=float(np.mean(OK)), collisions=float(np.mean(C)),
                avg_path_len=float(np.mean(PL)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=150000)
    ap.add_argument("--eval", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--algos", nargs="+", default=["ppo", "sac", "td3"])
    args = ap.parse_args()

    results_dir = os.path.join(_HERE, "results"); os.makedirs(results_dir, exist_ok=True)
    tb_dir = os.path.join(_HERE, "runs")
    summary = {}
    grand = time.time()

    for algo in args.algos:
        print(f"\n========== {algo.upper()} (L1 warm-start) ==========", flush=True)
        cfg = load_config(os.path.join(_CFG, f"{algo}.yaml"))
        cfg["env"]["curriculum"]["level"] = 1
        cfg["env"]["max_episode_steps"] = args.max_steps
        cfg["seed"] = args.seed
        save_dir = os.path.join(tb_dir, algo); os.makedirs(save_dir, exist_ok=True)
        monitor_path = os.path.join(save_dir, "monitor")
        env = make_env(cfg, args.seed, monitor_path)

        warm = os.path.join(_WARM, f"{algo}_final.zip")
        hp = dict(cfg.get(algo, {})); hp.pop("policy", None)
        if os.path.isfile(warm):
            print(f"[{algo}] warm-starting from {warm}", flush=True)
            model = ALGOS[algo].load(warm, env=env, tensorboard_log=tb_dir)
        else:
            kwargs = dict(policy="MlpPolicy", env=env, verbose=0, seed=args.seed,
                          tensorboard_log=tb_dir)
            if algo == "td3":
                std = float(hp.pop("action_noise_std", 0.1)); n = env.action_space.shape[0]
                kwargs["action_noise"] = NormalActionNoise(np.zeros(n), std * np.ones(n))
            kwargs.update(hp)
            model = ALGOS[algo](**kwargs)

        ckpt = CheckpointCallback(save_freq=max(args.timesteps // 4, 1),
                                  save_path=save_dir, name_prefix=algo)
        t0 = time.time()
        model.learn(total_timesteps=args.timesteps, callback=ckpt, tb_log_name=algo,
                    reset_num_timesteps=True, progress_bar=False)
        train_min = (time.time() - t0) / 60
        model.save(os.path.join(save_dir, f"{algo}_final"))
        env.close()

        m = evaluate(model, cfg, args.eval, args.seed, args.max_steps)
        m["train_minutes"] = round(train_min, 1)
        summary[algo] = m
        print(f"[{algo}] L1 eval: deliv={m['delivery_rate']*100:.0f}% "
              f"success={m['success_rate']*100:.0f}% coll/ep={m['collisions']:.2f} "
              f"reward={m['avg_reward']:.0f} ({train_min:.0f} min)", flush=True)

    meta = {"level": 1, "timesteps": args.timesteps, "eval_episodes": args.eval,
            "total_minutes": round((time.time() - grand) / 60, 1)}
    json.dump({"meta": meta, "results": summary},
              open(os.path.join(results_dir, "compare.json"), "w"), indent=2)
    lines = [f"# Training comparison (level L1 warm-start, {args.timesteps} steps, "
             f"{args.eval} eval episodes)\n",
             f"_total wall time: {meta['total_minutes']} min_\n",
             "| algo | delivery % | success % | coll/ep | path(m) | avg reward | train min |",
             "|------|-----------:|----------:|--------:|--------:|-----------:|----------:|"]
    for a in args.algos:
        s = summary.get(a, {})
        lines.append(f"| {a} | {s.get('delivery_rate',0)*100:.0f} | {s.get('success_rate',0)*100:.0f} "
                     f"| {s.get('collisions',0):.2f} | {s.get('avg_path_len',0):.1f} "
                     f"| {s.get('avg_reward',0):.0f} | {s.get('train_minutes',0):.0f} |")
    open(os.path.join(results_dir, "SUMMARY.md"), "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines), flush=True)
    print("L1_TRAINING_DONE", flush=True)


if __name__ == "__main__":
    main()
