"""Builds notebooks/experiments.ipynb (run once; deterministic, valid JSON).

Order (2D review + charts come BEFORE the heavy 3D GIFs, as requested):
  A. setup + smoke test
  B. optional short demo training (OFF by default)
  C. load trained models + numeric comparison (saved table, reward curves, eval)
  D. 2D top-down MAP REVIEW  -> robot paths on the rich map (corridor/rooms/arc)
  E. comparison CHARTS       -> grouped bars + reward mean/std
  F. 3D GIF gallery          -> rendered rollouts (LAST, heaviest)
  G. conclusion
"""
import json
import os

NB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments.ipynb")

cells = []
def md(s): cells.append(("markdown", s))
def code(s): cells.append(("code", s))

md("""# Delivery Robot RL — PPO vs SAC vs TD3 (sensor-only, PyBullet)

Apartment delivery robot on a **rich map**: a wide main corridor, several side
**rooms (ngõ ngách)**, and a curved **arc branch (hành lang vòng cung)**; lockers
are spread across all three areas. Sensors only (LiDAR + ToF + IMU + odometry) —
**no camera**. A global waypoint planner guides the reactive RL policy around
bends; pedestrian avoidance + motion smoothing are execution-time layers.

**Order:** A setup · B demo(off) · C numeric comparison · **D 2D map review** ·
**E charts** · **F 3D GIF gallery (last)** · G conclusion. The 2D maps and charts
appear *before* the 3D renders so you can compare the models without waiting on
the heavy GIFs.""")

# ----------------------------------------------------------------- A
md("## A1. Setup & version check")
code("""# (If needed) install dependencies:
# %pip install -r ../requirements.txt
import os, sys, time, glob, warnings
warnings.filterwarnings("ignore")

REPO_ROOT = None
for cand in [".", "..", os.path.join("..", "..")]:
    cand = os.path.abspath(cand)
    if os.path.isdir(os.path.join(cand, "delivery_rl")):
        REPO_ROOT = cand
        if cand not in sys.path:
            sys.path.insert(0, cand)
        break
assert REPO_ROOT, "could not locate the delivery_rl package"

from importlib.metadata import version, PackageNotFoundError
for pkg in ["gymnasium", "stable-baselines3", "pybullet", "torch",
            "pyyaml", "tensorboard", "numpy", "matplotlib", "imageio", "pillow"]:
    try:
        print(f"{pkg:18s} {version(pkg)}")
    except PackageNotFoundError:
        print(f"{pkg:18s} NOT INSTALLED")
print("repo root:", REPO_ROOT)""")

md("## A2. Smoke test — create env, run 300 random steps, print spaces")
code("""import numpy as np
from delivery_rl.configs.loader import load_config, default_config_path
from delivery_rl.envs.corridor_delivery_env import CorridorDeliveryEnv

cfg = load_config(default_config_path())
cfg["env"]["curriculum"]["level"] = 0
env = CorridorDeliveryEnv(config=cfg)
obs, info = env.reset(seed=0)
print("map style       :", cfg["env"]["world"]["map_style"])
print("observation_space:", env.observation_space)
print("action_space     :", env.action_space)
print("num lockers      :", env.num_lockers, "| obs shape:", obs.shape)
print("manifest (parcel->locker):", info["manifest"])
t0, n = time.time(), 300
for i in range(n):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs), f"obs out of bounds at step {i}"
    if term or trunc:
        obs, info = env.reset()
print(f"ran {n} random steps OK in {time.time()-t0:.1f}s (headless PyBullet DIRECT)")
env.close()""")

# ----------------------------------------------------------------- B
md("""## B. (Optional) Short demo training — PPO, SAC, TD3

**OFF by default** (`RUN_DEMO_TRAINING = False`) so the notebook is fast and
Section C uses the models already trained by `experiments_run.py`.""")
code("""from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise

RUN_DEMO_TRAINING = False
DEMO_TIMESTEPS = 4000
SEED, LEVEL = 0, 0

class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__(); self.x, self.y = [], []
    def _on_step(self):
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self.x.append(self.num_timesteps); self.y.append(ep["r"])
        return True

demo_curves, demo_models = {}, {}
if RUN_DEMO_TRAINING:
    def mk(max_steps=300):
        c = load_config(default_config_path()); c["env"]["curriculum"]["level"] = LEVEL
        c["env"]["max_episode_steps"] = max_steps
        e = Monitor(CorridorDeliveryEnv(config=c)); e.reset(seed=SEED); return e
    def build(algo, env):
        common = dict(policy="MlpPolicy", env=env, verbose=0, seed=SEED,
                      policy_kwargs=dict(net_arch=[128, 128]))
        if algo == "ppo": return PPO(n_steps=1024, batch_size=256, **common)
        if algo == "sac": return SAC(buffer_size=50000, learning_starts=500, batch_size=256, **common)
        n = env.action_space.shape[0]
        return TD3(buffer_size=50000, learning_starts=500, batch_size=256,
                   action_noise=NormalActionNoise(np.zeros(n), 0.1*np.ones(n)), **common)
    for algo in ["ppo", "sac", "td3"]:
        env = mk(); model = build(algo, env); cb = RewardLogger(); t0 = time.time()
        model.learn(total_timesteps=DEMO_TIMESTEPS, callback=cb, progress_bar=False)
        print(f"{algo}: {time.time()-t0:.1f}s, {len(cb.y)} episodes")
        demo_curves[algo] = (cb.x, cb.y); demo_models[algo] = model; env.close()
else:
    print("demo training skipped (RUN_DEMO_TRAINING=False) -> Section C uses trained models")""")

# ----------------------------------------------------------------- C
md("""---
## C. Trained models — numeric comparison

Loads `runs/<algo>/<algo>_final.zip` (train with
`python delivery_rl/experiments_run.py`).""")

md("### C1. Load models + saved comparison table")
code("""from delivery_rl.viz import list_trained_models
ALGOS = ["ppo", "sac", "td3"]
LOADERS = {"ppo": PPO, "sac": SAC, "td3": TD3}
paths = list_trained_models(REPO_ROOT)
models, source = {}, {}
for algo in ALGOS:
    if algo in paths:
        models[algo] = LOADERS[algo].load(paths[algo]); source[algo] = "trained"
    elif algo in demo_models:
        models[algo] = demo_models[algo]; source[algo] = "demo (B)"
print("models in use:", {a: source.get(a, "MISSING") for a in ALGOS})
summ = os.path.join(REPO_ROOT, "delivery_rl", "results", "SUMMARY.md")
print("\\n" + open(summ, encoding="utf-8").read() if os.path.isfile(summ)
      else "\\n(no results/SUMMARY.md yet — run experiments_run.py)")""")

md("### C2. Training reward curves (from monitor.csv)")
code("""import matplotlib.pyplot as plt
from delivery_rl.experiments_run import read_curve
plt.figure(figsize=(9, 4)); plotted = False
for algo in ALGOS:
    x, y = read_curve(os.path.join(REPO_ROOT, "delivery_rl", "runs", algo, "monitor"))
    if not y and algo in demo_curves: x, y = demo_curves[algo]
    if y:
        k = max(1, len(y)//40); ys = np.convolve(y, np.ones(k)/k, mode="valid")
        plt.plot(x[k-1:], ys, label=f"{algo.upper()} ({source.get(algo,'?')})", alpha=0.9)
        plotted = True
plt.xlabel("timestep"); plt.ylabel("episode reward (smoothed)")
plt.title("Training reward — PPO vs SAC vs TD3"); plt.grid(alpha=0.3)
if plotted: plt.legend()
plt.tight_layout(); plt.show()""")

md("### C3. Live evaluation table (deterministic)")
code("""def evaluate(model, level, episodes=10, seed=500, max_steps=700):
    c = load_config(default_config_path()); c["env"]["curriculum"]["level"] = level
    c["env"]["max_episode_steps"] = max_steps
    env = CorridorDeliveryEnv(config=c); R,S,D,C,OK = [],[],[],[],[]
    for ep in range(episodes):
        obs, info = env.reset(seed=seed+ep); n=info["num_parcels"]; done=False
        tot=steps=coll=dd=0
        while not done:
            a,_=model.predict(obs,deterministic=True)
            obs,r,term,trunc,info=env.step(a)
            tot+=r; steps+=1; coll+=int(info["collision"]); dd=info["deliveries_done"]
            done=term or trunc
        R.append(tot);S.append(steps);D.append(dd/max(n,1));C.append(coll);OK.append(int(dd>=n))
    env.close()
    return dict(avg_reward=float(np.mean(R)),avg_steps=float(np.mean(S)),
                delivery_rate=float(np.mean(D)),success_rate=float(np.mean(OK)),
                collisions=float(np.mean(C)))
EVAL_LEVEL = 0
results = {a: evaluate(models[a], EVAL_LEVEL) for a in ALGOS if a in models}
hdr = f"{'algo':6s}{'deliv%':>9s}{'reach%':>9s}{'coll/ep':>9s}{'avg_steps':>11s}{'avg_reward':>12s}"
print(f"(live eval, level L{EVAL_LEVEL})"); print(hdr); print("-"*len(hdr))
for a, s in results.items():
    print(f"{a:6s}{s['delivery_rate']*100:9.1f}{s['success_rate']*100:9.1f}"
          f"{s['collisions']:9.2f}{s['avg_steps']:11.1f}{s['avg_reward']:12.1f}")""")

# ----------------------------------------------------------------- D  (2D, BEFORE 3D)
md("""---
## D. 2D top-down MAP REVIEW (before any 3D)

Static top-down plots (matplotlib — light, never freezes). All three models are
sent to the **same destination locker** (red dot; dock point = red star) and
their driven paths overlaid, across **different areas of the map**: a corridor
locker, a room locker (ngõ ngách), and an arc-branch locker (vòng cung), plus
obstacle / pedestrian variants.""")
code("""from delivery_rl.analysis import (plot_scenarios_grid, benchmark, plot_benchmark_charts)
import matplotlib.pyplot as plt

# Main-corridor lockers, ordered by distance from the dock (near -> far). The
# models were trained to deliver to ANY corridor locker, so we compare them
# heading to a NEAR, a MID and a FAR destination, plus a far+obstacles variant.
def corridor_lockers_sorted():
    c = load_config(default_config_path()); e = CorridorDeliveryEnv(config=c); e.reset(seed=0)
    dock = e.scene.dock_pos[:2]
    north = [l for l in e.scene.lockers if l.side == "north"]
    north.sort(key=lambda l: (l.dock[0]-dock[0])**2 + (l.dock[1]-dock[1])**2)
    e.close(); return [l.id for l in north]
CORR = corridor_lockers_sorted()
print("corridor lockers (near->far):", CORR)
l_near, l_mid, l_far = CORR[0], CORR[len(CORR)//2], CORR[-1]

MAP_SCENARIOS = [
    {"name": f"corridor NEAR -> #{l_near}", "level": 1, "override": {"force_locker_id": l_near}},
    {"name": f"corridor MID -> #{l_mid}", "level": 1, "override": {"force_locker_id": l_mid}},
    {"name": f"corridor FAR -> #{l_far}", "level": 1, "override": {"force_locker_id": l_far}},
    {"name": f"FAR + obstacles -> #{l_far}", "level": 1,
     "override": {"force_locker_id": l_far, "num_obstacles": 4, "locker_sides": ["north"]}},
]
fig_dir = os.path.join(REPO_ROOT, "delivery_rl", "results", "figs"); os.makedirs(fig_dir, exist_ok=True)
fig, path_summary = plot_scenarios_grid(models, MAP_SCENARIOS, seed=2024,
                                        savepath=os.path.join(fig_dir, "map_review.png"))
plt.show()
print("\\npath length per model per map (reach = delivered to the locker):")
for s in path_summary:
    print(f"  {s['scenario']:26s} {s['algo'].upper():4s} "
          f"{'reach' if s['reached'] else 'MISS ':5s} {s['path_len']:5.1f} m / {s['steps']} steps")""")

md("### D2. Per-map route length (bar chart)")
code("""maps = list(dict.fromkeys(s["scenario"] for s in path_summary))
colors_d = {"ppo": "#1f77b4", "sac": "#ff7f0e", "td3": "#2ca02c"}
vals = {a: [next((s["path_len"] for s in path_summary if s["scenario"]==m and s["algo"]==a), 0.0)
            for m in maps] for a in ["ppo","sac","td3"]}
x = np.arange(len(maps)); w = 0.25
fig, ax = plt.subplots(figsize=(11, 4))
for i, a in enumerate(["ppo","sac","td3"]):
    ax.bar(x + (i-1)*w, vals[a], w, label=a.upper(), color=colors_d[a], alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels([m.replace(" -> ", "\\n-> ") for m in maps], fontsize=8)
ax.set_ylabel("path length (m)"); ax.set_title("Route length to the locker — per map")
ax.legend(); ax.grid(axis="y", alpha=0.3); plt.tight_layout(); plt.show()""")

# --------------------------------------------------- D3  interactive 2D picker
md("""### D3. 🎮 Pick a locker → 2D animation (interactive)

**This is the 2D test you choose.** Set `TEST_LOCKER_ID` to any locker id from
the table below and `TEST_ALGO` to the model, then run the next cell — it renders
a **2D top-down GIF** of that robot driving from the dock to the chosen locker
(red dot). No 3D needed. Add obstacles/pedestrians with the knobs if you like.

Tip: the **corridor (`north`) lockers are the reliably reachable ones**; room /
arc lockers are harder for the sensor-only policy.""")
code("""from delivery_rl.analysis import list_lockers, animate_2d_rollout
from IPython.display import Image, display
import pandas as _pd  # only for a tidy table; falls back to plain print if absent

_lk = list_lockers()
try:
    display(_pd.DataFrame(_lk))
except Exception:
    print("id | area  | pos | dist_from_dock")
    for d in _lk:
        print(f"  {d['id']:2d} | {d['area']:5s} | {d['pos']} | {d['dist_from_dock']}m")""")
code("""# ====== CHOOSE HERE ======
TEST_ALGO        = "ppo"     # "ppo" | "sac" | "td3"
TEST_LOCKER_ID   = 5          # any id from the table above
TEST_OBSTACLES   = 0          # e.g. 4 to drop static boxes in the corridor
TEST_PEDESTRIANS = 0          # e.g. 3 to add moving residents
TEST_SEED        = 2024
# =========================
assert TEST_ALGO in models, f"{TEST_ALGO} not loaded"
_gif = os.path.join(fig_dir, f"pick_{TEST_ALGO}_locker{TEST_LOCKER_ID}.gif")
_out = animate_2d_rollout(models[TEST_ALGO], TEST_LOCKER_ID, filename=_gif,
                          level=1, seed=TEST_SEED, max_steps=900,
                          num_obstacles=TEST_OBSTACLES, num_pedestrians=TEST_PEDESTRIANS,
                          fps=15, max_frames=110, label=TEST_ALGO.upper())
print(f"{TEST_ALGO.upper()} -> locker #{TEST_LOCKER_ID}: "
      f"reached={_out['reached']} | steps={_out['steps']} | "
      f"path={_out['path_len']:.1f} m | {_out['frames']} frames")
display(Image(filename=_gif))""")

# --------------------------------------------------- D4  all 3 models at once
md("""### D4. 🎮 All 3 models at once → one 2D animation

Same idea as D3 but runs **PPO, SAC and TD3 together** to the locker you choose
and overlays all three robots in a single 2D GIF (each its own colour), so you
can watch them race to the same red-dot locker. The title shows ✓/✗ per model.""")
code("""from delivery_rl.analysis import animate_2d_compare
from IPython.display import Image, display

# ====== CHOOSE HERE ======
CMP_LOCKER_ID   = 5          # any id from the D3 table
CMP_OBSTACLES   = 0          # e.g. 4 static boxes
CMP_PEDESTRIANS = 0          # e.g. 3 moving residents
CMP_SEED        = 2024
# =========================
_cgif = os.path.join(fig_dir, f"compare_locker{CMP_LOCKER_ID}.gif")
_cout = animate_2d_compare(models, CMP_LOCKER_ID, filename=_cgif,
                           level=1, seed=CMP_SEED, max_steps=900,
                           num_obstacles=CMP_OBSTACLES, num_pedestrians=CMP_PEDESTRIANS,
                           fps=15, max_frames=110)
print(f"locker #{CMP_LOCKER_ID} — per model:")
for a, s in _cout["per_model"].items():
    print(f"  {a.upper():4s} reached={s['reached']} steps={s['steps']}")
display(Image(filename=_cgif))""")

# ----------------------------------------------------------------- E  (charts, BEFORE 3D)
md("""---
## E. Comparison CHARTS (before any 3D)

Quantitative comparison over many episodes: delivery rate, reward, steps,
collisions, path length, reach rate; plus reward mean ± std.""")
code("""bench = benchmark(models, level=1, override={"locker_sides": ["north"]},
                  episodes=12, seed0=300)
fig = plot_benchmark_charts(bench, "(random main-corridor locker, 12 episodes)",
                            savepath=os.path.join(fig_dir, "benchmark.png"))
plt.show()
hdr = f"{'algo':6s}{'deliv%':>8s}{'reach%':>8s}{'reward':>9s}{'steps':>8s}{'path(m)':>9s}{'coll':>7s}"
print(hdr); print("-"*len(hdr))
for a, v in bench.items():
    print(f"{a:6s}{v['delivery_rate']*100:8.0f}{v['reach_rate']*100:8.0f}"
          f"{v['avg_reward']:9.1f}{v['avg_steps']:8.0f}{v['avg_path_len']:9.1f}{v['collisions']:7.2f}")""")
code("""labels = [a.upper() for a in bench]
means = [bench[a]["avg_reward"] for a in bench]; stds = [bench[a]["reward_std"] for a in bench]
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(labels, means, yerr=stds, capsize=8, color=[colors_d[a] for a in bench], alpha=0.85)
ax.set_ylabel("episode reward"); ax.set_title("Reward (mean ± std) — PPO vs SAC vs TD3")
ax.grid(axis="y", alpha=0.3); plt.tight_layout(); plt.show()""")

# ----------------------------------------------------------------- F  (3D, LAST)
md("""---
## F. 3D GIF gallery (rendered — shown LAST)

Heaviest section: animated rollouts from the synthetic overhead camera. Prefers
pre-generated GIFs (`make_gallery.py`). If the notebook feels heavy, set
`SHOW_GIFS_INLINE = False` to list paths instead and open them from
`results/gifs/`.""")
code("""from IPython.display import Image, display, Markdown
from delivery_rl.viz import default_scenarios, record_rollout_gif
GIF_DIR = os.path.join(REPO_ROOT, "delivery_rl", "results", "gifs"); os.makedirs(GIF_DIR, exist_ok=True)
SEEDS = {"clear": 2024, "obstacles": 2024, "pedestrians": 7}
gallery = {}
for algo in ALGOS:
    if algo not in models: continue
    gallery[algo] = []
    for sc in default_scenarios():
        name, level, override = sc["name"], sc["level"], sc["override"]
        path = os.path.join(GIF_DIR, f"{algo}_{name}.gif")
        if not (os.path.isfile(path) and os.path.getsize(path) > 0):
            record_rollout_gif(models[algo], level=level, filename=path, max_steps=1000,
                               fps=12, max_frames=90, seed=SEEDS.get(name, 2024),
                               scenario_override=override, label=algo.upper())
        gallery[algo].append((name, level, path))
        print(f"[{algo}] {name} ready")
print("GIF dir:", GIF_DIR)""")
code("""SHOW_GIFS_INLINE = True
for algo in ALGOS:
    if algo not in gallery: continue
    display(Markdown(f"#### {algo.upper()}  *(source: {source.get(algo,'?')})*"))
    for name, level, path in gallery[algo]:
        ok = os.path.isfile(path) and os.path.getsize(path) > 0
        if SHOW_GIFS_INLINE and ok:
            display(Markdown(f"**{name}**")); display(Image(filename=path))
        elif ok:
            print(f"  {name:12s} -> {path}  ({os.path.getsize(path)//1024} KB)")
        else:
            print(f"  {name:12s} -> (GIF missing)")""")

# ----------------------------------------------------------------- G
md("### G. Conclusion — best model per metric")
code("""metrics = {"delivery rate": ("delivery_rate", True), "reach rate": ("success_rate", True),
           "collisions/ep": ("collisions", False), "avg reward": ("avg_reward", True),
           "avg steps": ("avg_steps", False)}
if results:
    print("Best model per metric (live eval, level L%d):" % EVAL_LEVEL)
    for label,(key,higher) in metrics.items():
        best=(max if higher else min)(results, key=lambda a: results[a][key])
        print(f"  {label:14s}: {best.upper():4s} ({results[best][key]:.2f})")
    overall=max(results, key=lambda a:(results[a]['delivery_rate'], results[a]['avg_reward']))
    print(f"\\n  OVERALL  : {overall.upper()} (deliv {results[overall]['delivery_rate']*100:.0f}%, "
          f"reward {results[overall]['avg_reward']:.0f})")
else:
    print("No results — train models first.")""")

nb = {"cells": [], "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}
for ctype, src in cells:
    cell = {"cell_type": ctype, "metadata": {}, "source": src.splitlines(keepends=True)}
    if ctype == "code":
        cell["execution_count"] = None; cell["outputs"] = []
    nb["cells"].append(cell)
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote", NB_PATH, "| cells:", len(cells))
