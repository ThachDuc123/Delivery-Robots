"""Real training + evaluation driver: PPO vs SAC vs TD3 on one shared env.

Trains each algorithm for the same number of timesteps on the same env/seed,
logs TensorBoard + a per-algo monitor.csv (for reward curves), saves the final
model, then evaluates deterministically. All artifacts go under runs/ and
results/ so the notebook / chat can show the real comparison.

Usage:
    python delivery_rl/experiments_run.py                 # defaults below
    python delivery_rl/experiments_run.py --timesteps 100000 --level 1 --eval 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

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
_CONFIG_DIR = os.path.join(_HERE, "configs")


def make_env(config: dict, seed: int, monitor_path: str) -> Monitor:
    env = CorridorDeliveryEnv(config=config)
    env = Monitor(env, filename=monitor_path)
    env.reset(seed=seed)
    return env


def build_model(algo: str, env, config: dict, seed: int, tb_dir: str):
    hp = dict(config.get(algo, {}))
    policy = hp.pop("policy", "MlpPolicy")
    kwargs = dict(policy=policy, env=env, verbose=0, seed=seed, tensorboard_log=tb_dir)
    if algo == "td3":
        std = float(hp.pop("action_noise_std", 0.1))
        n = env.action_space.shape[0]
        kwargs["action_noise"] = NormalActionNoise(mean=np.zeros(n), sigma=std * np.ones(n))
    kwargs.update(hp)
    return ALGOS[algo](**kwargs)


def evaluate(model, config: dict, episodes: int, seed: int) -> dict:
    env = CorridorDeliveryEnv(config=config)
    R, S, D, C, OK = [], [], [], [], []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + 1000 + ep)
        n_parcels, done = info["num_parcels"], False
        tot = steps = coll = dd = 0
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            tot += r; steps += 1; coll += int(info["collision"]); dd = info["deliveries_done"]
            done = term or trunc
        R.append(tot); S.append(steps); D.append(dd / max(n_parcels, 1))
        C.append(coll); OK.append(int(info["is_success"]))
    env.close()
    return dict(avg_reward=float(np.mean(R)), avg_steps=float(np.mean(S)),
                delivery_rate=float(np.mean(D)), success_rate=float(np.mean(OK)),
                collisions_per_ep=float(np.mean(C)))


def read_curve(monitor_path: str):
    """Return (cum_timesteps, episode_rewards) from a SB3 monitor.csv."""
    path = monitor_path + ".monitor.csv"
    xs, ys = [], []
    if not os.path.isfile(path):
        return xs, ys
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln for ln in f.read().splitlines() if ln and not ln.startswith("#")]
    if not lines:
        return xs, ys
    header = lines[0].split(",")
    try:
        ri, li = header.index("r"), header.index("l")
    except ValueError:
        return xs, ys
    cum = 0
    for ln in lines[1:]:
        parts = ln.split(",")
        try:
            r = float(parts[ri]); l = int(float(parts[li]))
        except (ValueError, IndexError):
            continue
        cum += l
        xs.append(cum); ys.append(r)
    return xs, ys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=100000)
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--eval", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--algos", nargs="+", default=["ppo", "sac", "td3"])
    args = ap.parse_args()

    results_dir = os.path.join(_HERE, "results")
    os.makedirs(results_dir, exist_ok=True)
    tb_dir = os.path.join(_HERE, "runs")

    summary, curves = {}, {}
    grand_t0 = time.time()
    for algo in args.algos:
        print(f"\n========== {algo.upper()} ==========", flush=True)
        try:
            config = load_config(os.path.join(_CONFIG_DIR, f"{algo}.yaml"))
            config["env"]["curriculum"]["level"] = args.level
            config["env"]["max_episode_steps"] = args.max_steps
            config["seed"] = args.seed

            save_dir = os.path.join(tb_dir, algo)
            os.makedirs(save_dir, exist_ok=True)
            monitor_path = os.path.join(save_dir, "monitor")

            env = make_env(config, args.seed, monitor_path)
            model = build_model(algo, env, config, args.seed, tb_dir)
            ckpt = CheckpointCallback(save_freq=max(args.timesteps // 4, 1),
                                      save_path=save_dir, name_prefix=algo)
            t0 = time.time()
            model.learn(total_timesteps=args.timesteps, callback=ckpt, tb_log_name=algo,
                        progress_bar=False)
            train_s = time.time() - t0
            final = os.path.join(save_dir, f"{algo}_final")
            model.save(final)
            env.close()
            print(f"[{algo}] trained {args.timesteps} steps in {train_s/60:.1f} min "
                  f"-> {final}.zip", flush=True)

            metrics = evaluate(model, config, args.eval, args.seed)
            metrics["train_minutes"] = round(train_s / 60, 2)
            metrics["timesteps"] = args.timesteps
            summary[algo] = metrics
            curves[algo] = read_curve(monitor_path)
            print(f"[{algo}] eval: deliv={metrics['delivery_rate']*100:.1f}% "
                  f"success={metrics['success_rate']*100:.1f}% "
                  f"coll/ep={metrics['collisions_per_ep']:.2f} "
                  f"avg_reward={metrics['avg_reward']:.1f}", flush=True)
        except Exception as e:
            summary[algo] = {"error": f"{type(e).__name__}: {e}"}
            traceback.print_exc()

    meta = {"level": args.level, "timesteps": args.timesteps, "eval_episodes": args.eval,
            "max_steps": args.max_steps, "seed": args.seed,
            "total_minutes": round((time.time() - grand_t0) / 60, 2)}
    with open(os.path.join(results_dir, "compare.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": summary}, f, indent=2)
    with open(os.path.join(results_dir, "curves.json"), "w", encoding="utf-8") as f:
        json.dump(curves, f)

    # human-readable table
    lines = [f"# Training comparison (level L{args.level}, {args.timesteps} steps, "
             f"{args.eval} eval episodes)\n",
             f"_total wall time: {meta['total_minutes']} min_\n",
             "| algo | delivery % | success % | coll/ep | avg steps | avg reward | train min |",
             "|------|-----------:|----------:|--------:|----------:|-----------:|----------:|"]
    for algo in args.algos:
        s = summary.get(algo, {})
        if "error" in s:
            lines.append(f"| {algo} | ERROR: {s['error']} |||||| |")
        else:
            lines.append(f"| {algo} | {s['delivery_rate']*100:.1f} | {s['success_rate']*100:.1f} "
                         f"| {s['collisions_per_ep']:.2f} | {s['avg_steps']:.0f} "
                         f"| {s['avg_reward']:.1f} | {s['train_minutes']:.1f} |")
    table = "\n".join(lines) + "\n"
    with open(os.path.join(results_dir, "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write(table)
    print("\n" + table, flush=True)
    print("ARTIFACTS: runs/<algo>/<algo>_final.zip, results/compare.json, "
          "results/curves.json, results/SUMMARY.md", flush=True)
    print("EXPERIMENTS_DONE", flush=True)


if __name__ == "__main__":
    main()
