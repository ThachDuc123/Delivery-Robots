"""Evaluate one trained policy on CorridorDeliveryEnv and print the metrics.

Usage:
    python delivery_rl/eval.py --algo ppo --model delivery_rl/runs/ppo/ppo_final.zip
    python delivery_rl/eval.py --algo sac --episodes 20 --level 0
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from stable_baselines3 import PPO, SAC, TD3

from delivery_rl.configs.loader import load_config
from delivery_rl.envs.corridor_delivery_env import CorridorDeliveryEnv

ALGOS = {"ppo": PPO, "sac": SAC, "td3": TD3}
_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")


def evaluate(model, env, episodes: int, seed: int = 0) -> dict:
    rewards, steps, deliv_rate, collisions, successes = [], [], [], [], []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        total_r, n_steps, n_coll = 0.0, 0, 0
        n_parcels = info["num_parcels"]
        done_count = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            total_r += r
            n_steps += 1
            n_coll += int(info["collision"])
            done_count = info["deliveries_done"]
            done = term or trunc
        rewards.append(total_r)
        steps.append(n_steps)
        deliv_rate.append(done_count / max(n_parcels, 1))
        collisions.append(n_coll)
        successes.append(int(info["is_success"]))
    return {
        "episodes": episodes,
        "delivery_rate": float(np.mean(deliv_rate)),
        "success_rate": float(np.mean(successes)),
        "collisions_per_ep": float(np.mean(collisions)),
        "avg_steps": float(np.mean(steps)),
        "avg_reward": float(np.mean(rewards)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a policy on CorridorDeliveryEnv")
    parser.add_argument("--algo", required=True, choices=list(ALGOS))
    parser.add_argument("--model", required=True, help="path to the saved .zip model")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--level", type=int, default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config or os.path.join(_CONFIG_DIR, f"{args.algo}.yaml"))
    if args.level is not None:
        config["env"]["curriculum"]["level"] = args.level

    env = CorridorDeliveryEnv(config=config)
    model = ALGOS[args.algo].load(args.model)
    stats = evaluate(model, env, args.episodes, seed=int(config.get("seed", 0)))
    env.close()

    print(f"\n=== Eval: {args.algo}  (level {config['env']['curriculum']['level']}, "
          f"{stats['episodes']} episodes) ===")
    print(f"  delivery rate    : {stats['delivery_rate'] * 100:5.1f} %")
    print(f"  success rate     : {stats['success_rate'] * 100:5.1f} %")
    print(f"  collisions / ep  : {stats['collisions_per_ep']:.2f}")
    print(f"  avg steps        : {stats['avg_steps']:.1f}")
    print(f"  avg reward       : {stats['avg_reward']:.1f}")


if __name__ == "__main__":
    main()
