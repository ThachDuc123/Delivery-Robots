"""Train PPO/SAC/TD3 to deliver to ANY main-corridor locker (warm-started).

Strategy ("làm thật tốt"):
  * Warm-start each algo from its L0 model (which already drives the corridor and
    returns to the dock), so we keep that competence and only broaden the set of
    reachable destinations.
  * Restrict targets to the main-corridor ("north") lockers via the task override
    ``locker_sides=["north"]`` -- the reliably reachable destinations -- and pick
    a RANDOM one each episode (curriculum level 1). This teaches the robot to go
    to many different corridor lockers, which is what the 2D map review compares.
  * Evaluate by forcing EACH corridor locker in turn so the report reflects the
    real per-destination reach rate (not just the random average).

Artifacts: runs/<algo>/<algo>_final.zip (overwrites), results/SUMMARY.md,
results/compare.json, results/curves.json, results/train_log.txt.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from delivery_rl.configs.loader import load_config, default_config_path
from delivery_rl.envs.corridor_delivery_env import CorridorDeliveryEnv

ALGOS = {"ppo": PPO, "sac": SAC, "td3": TD3}
POOL = {"locker_sides": ["north"]}          # main-corridor lockers only
TIMESTEPS = {"ppo": 200_000, "sac": 120_000, "td3": 120_000}
MAX_STEPS = 700
SEED = 0
WARM_DIR = os.path.join(_HERE, "_L0newmap_backup")


def make_env(level=1, seed=SEED, monitor=None):
    cfg = load_config(default_config_path())
    cfg["env"]["curriculum"]["level"] = level
    cfg["env"]["max_episode_steps"] = MAX_STEPS
    cfg["env"]["scenario_override"] = dict(POOL)
    env = CorridorDeliveryEnv(config=cfg)
    env = Monitor(env, filename=monitor)
    env.reset(seed=seed)
    return env


def evaluate_each_locker(model, episodes_per=2):
    """Force every corridor locker; return aggregate + per-locker reach."""
    cfg = load_config(default_config_path())
    env0 = CorridorDeliveryEnv(config=cfg); env0.reset(seed=0)
    north = [l.id for l in env0.scene.lockers if l.side == "north"]
    env0.close()

    R, S, D, C, OK, PL = [], [], [], [], [], []
    per = {}
    for lid in north:
        reached_count = 0
        for k in range(episodes_per):
            cfg = load_config(default_config_path())
            cfg["env"]["max_episode_steps"] = 900
            cfg["env"]["scenario_override"] = {"force_locker_id": lid}
            env = CorridorDeliveryEnv(config=cfg)
            obs, info = env.reset(seed=100 + k)
            n = info["num_parcels"]; done = False
            tot = steps = coll = dd = 0
            px, py, _ = env.robot.get_pose(); plen = 0.0
            while not done:
                a, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(a)
                tot += r; steps += 1; coll += int(info["collision"]); dd = info["deliveries_done"]
                nx, ny, _ = env.robot.get_pose(); plen += float(np.hypot(nx - px, ny - py)); px, py = nx, ny
                done = term or trunc
            env.close()
            reached = int(dd >= n)
            reached_count += reached
            R.append(tot); S.append(steps); D.append(dd / max(n, 1))
            C.append(coll); OK.append(reached); PL.append(plen)
        per[lid] = reached_count / episodes_per
    return {
        "avg_reward": float(np.mean(R)), "avg_steps": float(np.mean(S)),
        "delivery_rate": float(np.mean(D)), "reach_rate": float(np.mean(OK)),
        "collisions": float(np.mean(C)), "avg_path_len": float(np.mean(PL)),
        "per_locker_reach": per,
    }


def read_curve(monitor_path):
    path = monitor_path + ".monitor.csv"
    xs, ys = [], []
    if not os.path.isfile(path):
        return xs, ys
    lines = [l for l in open(path, encoding="utf-8").read().splitlines()
             if l and not l.startswith("#")]
    if not lines:
        return xs, ys
    hdr = lines[0].split(","); ri, li = hdr.index("r"), hdr.index("l")
    cum = 0
    for ln in lines[1:]:
        p = ln.split(",")
        try:
            cum += int(float(p[li])); xs.append(cum); ys.append(float(p[ri]))
        except (ValueError, IndexError):
            pass
    return xs, ys


def build(algo, env, warm_path):
    cfg = load_config(os.path.join(_HERE, "configs", f"{algo}.yaml"))
    hp = dict(cfg.get(algo, {})); hp.pop("policy", None)
    if os.path.isfile(warm_path):
        model = ALGOS[algo].load(warm_path, env=env)
        # refresh exploration/lr for the new (broader) objective
        return model, True
    common = dict(policy="MlpPolicy", env=env, verbose=0, seed=SEED,
                  policy_kwargs=dict(net_arch=[256, 256]))
    if algo == "td3":
        n = env.action_space.shape[0]
        common["action_noise"] = NormalActionNoise(np.zeros(n), 0.1 * np.ones(n))
    return ALGOS[algo](**common), False


def main():
    results, curves = {}, {}
    log = open(os.path.join(_HERE, "results", "train_log.txt"), "w", encoding="utf-8")
    def out(s):
        print(s, flush=True); log.write(s + "\n"); log.flush()

    grand = time.time()
    for algo in ("ppo", "sac", "td3"):
        out(f"\n========== {algo.upper()} (corridor pool, warm-start) ==========")
        save_dir = os.path.join(_HERE, "runs", algo)
        os.makedirs(save_dir, exist_ok=True)
        mon = os.path.join(save_dir, "monitor")
        env = make_env(monitor=mon)
        warm = os.path.join(WARM_DIR, f"{algo}_final.zip")
        model, warmed = build(algo, env, warm)
        out(f"[{algo}] warm-start={'yes' if warmed else 'no'} steps={TIMESTEPS[algo]}")
        ckpt = CheckpointCallback(save_freq=max(TIMESTEPS[algo] // 3, 1),
                                  save_path=save_dir, name_prefix=algo)
        t0 = time.time()
        model.learn(total_timesteps=TIMESTEPS[algo], callback=ckpt,
                    tb_log_name=algo, progress_bar=False, reset_num_timesteps=False)
        mins = (time.time() - t0) / 60
        model.save(os.path.join(save_dir, f"{algo}_final"))
        env.close()
        m = evaluate_each_locker(model)
        m["train_minutes"] = round(mins, 1)
        results[algo] = m
        curves[algo] = read_curve(mon)
        out(f"[{algo}] deliv={m['delivery_rate']*100:.0f}% reach={m['reach_rate']*100:.0f}% "
            f"coll/ep={m['collisions']:.2f} reward={m['avg_reward']:.0f} ({mins:.0f} min)")
        out(f"[{algo}] per-locker reach: {m['per_locker_reach']}")

    rd = os.path.join(_HERE, "results")
    json.dump({"results": results}, open(os.path.join(rd, "compare.json"), "w"), indent=2)
    json.dump(curves, open(os.path.join(rd, "curves.json"), "w"))
    lines = [f"# Corridor-delivery comparison (random main-corridor locker, warm-start)\n",
             f"_total wall time: {(time.time()-grand)/60:.1f} min_\n",
             "| algo | delivery % | reach % | coll/ep | path(m) | avg reward | train min |",
             "|------|-----------:|--------:|--------:|--------:|-----------:|----------:|"]
    for a in ("ppo", "sac", "td3"):
        s = results[a]
        lines.append(f"| {a} | {s['delivery_rate']*100:.0f} | {s['reach_rate']*100:.0f} "
                     f"| {s['collisions']:.2f} | {s['avg_path_len']:.1f} "
                     f"| {s['avg_reward']:.0f} | {s['train_minutes']:.0f} |")
    open(os.path.join(rd, "SUMMARY.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    out("\n" + "\n".join(lines))
    out("CORRIDOR_TRAINING_DONE")
    log.close()


if __name__ == "__main__":
    main()
