"""Train PPO / SAC / TD3 on the CorridorDeliveryEnv (shared env + config).

Usage (from the repo root or this folder):
    python delivery_rl/train.py --algo ppo
    python delivery_rl/train.py --algo sac --timesteps 50000 --level 0
    python delivery_rl/train.py --algo td3 --config delivery_rl/configs/td3.yaml

Checkpoints + final model are saved under runs/<algo>/ and TensorBoard logs under
runs/ (run `tensorboard --logdir delivery_rl/runs`).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# make `delivery_rl` importable no matter where this is launched from
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from delivery_rl.configs.loader import load_config
from delivery_rl.envs.corridor_delivery_env import CorridorDeliveryEnv

ALGOS = {"ppo": PPO, "sac": SAC, "td3": TD3}
_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")


def make_env(config: dict, seed: int) -> Monitor:
    env = CorridorDeliveryEnv(config=config)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def build_model(algo: str, env, config: dict, seed: int, tb_dir: str):
    hp = dict(config.get(algo, {}))
    policy = hp.pop("policy", "MlpPolicy")
    kwargs = dict(policy=policy, env=env, verbose=1, seed=seed, tensorboard_log=tb_dir)
    if algo == "td3":
        std = float(hp.pop("action_noise_std", 0.1))
        n = env.action_space.shape[0]
        kwargs["action_noise"] = NormalActionNoise(mean=np.zeros(n), sigma=std * np.ones(n))
    kwargs.update(hp)
    return ALGOS[algo](**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO/SAC/TD3 on CorridorDeliveryEnv")
    parser.add_argument("--algo", required=True, choices=list(ALGOS))
    parser.add_argument("--config", default=None, help="override config path")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--level", type=int, default=None, help="override curriculum level")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config_path = args.config or os.path.join(_CONFIG_DIR, f"{args.algo}.yaml")
    config = load_config(config_path)
    if args.level is not None:
        config["env"]["curriculum"]["level"] = args.level
    if args.seed is not None:
        config["seed"] = args.seed
    seed = int(config.get("seed", 0))

    base = os.path.dirname(os.path.abspath(__file__))
    tb_dir = os.path.join(base, config["train"]["tensorboard_dir"])
    save_dir = os.path.join(base, config["train"]["save_dir"], args.algo)
    os.makedirs(save_dir, exist_ok=True)

    env = make_env(config, seed)
    model = build_model(args.algo, env, config, seed, tb_dir)

    ckpt = CheckpointCallback(save_freq=int(config["train"]["checkpoint_freq"]),
                              save_path=save_dir, name_prefix=args.algo)
    total = args.timesteps if args.timesteps is not None else int(config["train"]["total_timesteps"])
    print(f"[train] algo={args.algo} level={config['env']['curriculum']['level']} "
          f"timesteps={total} seed={seed}")
    model.learn(total_timesteps=total, callback=ckpt, tb_log_name=args.algo)
    final_path = os.path.join(save_dir, f"{args.algo}_final")
    model.save(final_path)
    env.close()
    print(f"[train] saved -> {final_path}.zip")


if __name__ == "__main__":
    main()
