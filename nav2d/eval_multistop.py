"""Evaluate the multi-stop LSTM policy by running it DIRECTLY in the env.

Because MultiStopEnv is the deployment loop (planner + goal-transition inside),
there is no external runner: we just load the policy + its VecNormalize stats and
step the env, carrying the LSTM hidden state across the trip. Reports full-trip
(returned to dock) rate, average stops delivered, and collision rate, per map.
"""
from __future__ import annotations
import argparse, os, sys, collections
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from multistop_env import MultiStopEnv

CFG = dict(n_lidar=24, lidar_range=5.0, lookahead=1.6, max_steps=2000,
           grace_steps=18, collision_grace=25, reverse_frac=0.4)


def load(model_path, vec_path):
    m = RecurrentPPO.load(model_path)
    v = VecNormalize.load(vec_path, DummyVecEnv([lambda: MultiStopEnv(config=CFG)]))
    mean = v.obs_rms.mean.astype(np.float32); var = v.obs_rms.var.astype(np.float32)
    norm = lambda o: np.clip((o - mean) / np.sqrt(var + v.epsilon), -v.clip_obs, v.clip_obs).astype(np.float32)
    return m, norm


def run_episode(m, norm, env, seed, options=None):
    o, info = env.reset(seed=seed, options=options)
    state = None; es = np.ones(1, bool); done = False
    stops = 0; dock = False; coll = False; steps = 0
    while not done:
        a, state = m.predict(norm(o)[None], state=state, episode_start=es, deterministic=True)
        es = np.zeros(1, bool)
        o, r, t, tr, inf = env.step(a[0]); steps += 1
        stops = max(stops, inf["stops_done"]); dock = dock or inf["arrived_dock"]
        coll = coll or inf["collision"]; done = t or tr
    return {"stops": stops, "dock": dock, "coll": coll, "steps": steps,
            "stops_total": env.stops_total, "order": info["order"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(_HERE, "runs", "ms_lstm"))
    ap.add_argument("--episodes", type=int, default=20)
    args = ap.parse_args()
    m, norm = load(args.model, args.model + "_vecnorm.pkl")
    env = MultiStopEnv(config=CFG)
    print(f"=== Multi-stop LSTM eval ({args.episodes} trips/map) ===")
    for name in env.map_names:
        dock = 0; sd = 0; tot_s = 0; coll = 0
        for i in range(args.episodes):
            r = run_episode(m, norm, env, seed=9000 + i, options={"map": name})
            dock += int(r["dock"]); sd += r["stops"]; tot_s += r["stops_total"]; coll += int(r["coll"])
            # stops_total includes the return-to-dock leg; deliveries = stops_total-1
        n = args.episodes
        print(f"  {name}: full-trip(return dock) {dock/n*100:4.0f}%  "
              f"stops {sd}/{tot_s}  collision {coll/n*100:4.0f}%")
    print("EVAL_DONE")


if __name__ == "__main__":
    main()
