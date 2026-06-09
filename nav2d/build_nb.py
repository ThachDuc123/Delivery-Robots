"""Builds nav2d/experiments_2d.ipynb (run once; deterministic, valid JSON).

The notebook loads the trained policies, evaluates them on UNSEEN maps across all
corridor styles, draws the top-down path grid, shows the recorded GIFs inline,
and compares feed-forward PPO vs RecurrentPPO (LSTM). Pure 2D, no PyBullet.
"""
import json
import os

NB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments_2d.ipynb")
cells = []
def md(s): cells.append(("markdown", s))
def code(s): cells.append(("code", s))

md("""# nav2d — Sensor-only 2D navigation: test results & GIFs

The robot navigates **using only a LiDAR fan + relative goal bearing — no map,
no global planner**. Every episode is a *new* random corridor (straight / L-turn
/ S-curve / arc / U-turn / niches), and the task is a **round trip**: reach the
goal, then return to start. This notebook:

1. loads the trained policy/policies,
2. evaluates them on **unseen maps** (seeds never trained on),
3. draws the top-down **path grid**,
4. shows the **GIFs** of the robot driving each corridor type,
5. compares **PPO** vs **RecurrentPPO (LSTM)**.

All 2D, headless, light — runs in well under a minute (uses pre-trained models +
pre-recorded GIFs).""")

md("## 1. Setup")
code("""import os, sys, glob
sys.path.insert(0, os.path.abspath("."))   # nav2d/
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display, Markdown

from nav_env import Nav2DEnv
from world2d import STYLES
print("styles:", STYLES)
HERE = os.path.abspath(".")
RUNS = os.path.join(HERE, "runs"); RES = os.path.join(HERE, "results")
print("has PPO model:", os.path.isfile(os.path.join(RUNS, "ppo_nav2d.zip")))
print("has Recurrent model:", os.path.isfile(os.path.join(RUNS, "recurrent_nav2d.zip")))""")

md("""## 2. Load trained policies

`ppo_nav2d` = feed-forward PPO + frame-stack(4); `recurrent_nav2d` =
RecurrentPPO (LSTM). Each ships with its VecNormalize stats so observations are
normalised exactly as in training.""")
code("""from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

ENV_CFG = dict(n_lidar=24, lidar_range=5.0, max_steps=800, round_trip=True)
FRAME_STACK = 4

def load_vecnorm(path, frame_stack):
    venv = DummyVecEnv([lambda: Nav2DEnv(config=ENV_CFG)])
    if frame_stack:
        venv = VecFrameStack(venv, frame_stack)
    venv = VecNormalize.load(path, venv); venv.training = False; venv.norm_reward = False
    return venv

def normalizer(venv):
    mean, var = venv.obs_rms.mean.astype(np.float32), venv.obs_rms.var.astype(np.float32)
    eps, clip = venv.epsilon, venv.clip_obs
    return lambda x: np.clip((x-mean)/np.sqrt(var+eps), -clip, clip).astype(np.float32)

policies = {}   # name -> dict(model, norm, recurrent, frame_stack)
ppo_zip = os.path.join(RUNS, "ppo_nav2d.zip")
if os.path.isfile(ppo_zip):
    vn = load_vecnorm(os.path.join(RUNS, "ppo_nav2d_vecnorm.pkl"), FRAME_STACK)
    policies["PPO"] = dict(model=PPO.load(ppo_zip), norm=normalizer(vn),
                           recurrent=False, frame_stack=FRAME_STACK)
rec_zip = os.path.join(RUNS, "recurrent_nav2d.zip")
if os.path.isfile(rec_zip):
    from sb3_contrib import RecurrentPPO
    vn = load_vecnorm(os.path.join(RUNS, "recurrent_nav2d_vecnorm.pkl"), 0)
    policies["RecurrentPPO"] = dict(model=RecurrentPPO.load(rec_zip), norm=normalizer(vn),
                                    recurrent=True, frame_stack=0)
print("loaded policies:", list(policies))""")

md("## 3. Rollout helper (handles both frame-stack and LSTM)")
code("""def run_episode(pol, seed, style, max_steps=800, record_trail=False):
    env = Nav2DEnv(config=ENV_CFG)
    o, info = env.reset(seed=seed, options={"style": style})
    d = o.shape[0]
    stack = np.tile(o, pol["frame_stack"]) if pol["frame_stack"] else None
    state, estart = None, np.ones(1, dtype=bool)
    done = steps = 0; rg = rt = coll = False; done = False
    while not done and steps < max_steps:
        if pol["recurrent"]:
            a, state = pol["model"].predict(pol["norm"](o)[None], state=state,
                                            episode_start=estart, deterministic=True)
            estart = np.zeros(1, dtype=bool); a = a[0]
        else:
            a, _ = pol["model"].predict(pol["norm"](stack), deterministic=True)
        o, r, term, trunc, info = env.step(a)
        if pol["frame_stack"]:
            stack = np.concatenate([stack[d:], o])
        steps += 1; rg = rg or info["reached_goal"]; rt = rt or info["round_trip"]
        coll = coll or info["collision"]; done = term or trunc
    out = dict(reached_goal=rg, round_trip=rt, collision=coll, steps=steps)
    if record_trail:
        out["env"] = env
    return out
print("rollout helper ready")""")

md("## 4. Evaluation table on unseen maps")
code("""EP = 20
rows = {}
for name, pol in policies.items():
    agg = {}
    for style in STYLES:
        rg = rt = coll = stp = 0
        for sd in range(EP):
            res = run_episode(pol, seed=9000+sd, style=style)
            rg += res["reached_goal"]; rt += res["round_trip"]
            coll += res["collision"]; stp += res["steps"]
        agg[style] = (rg/EP, rt/EP, coll/EP, stp/EP)
    rows[name] = agg

for name, agg in rows.items():
    print(f"\\n=== {name} ({EP} unseen maps/style) ===")
    print(f"  {'style':9s}{'reach%':>8s}{'round%':>8s}{'coll%':>7s}{'steps':>7s}")
    tr=tg=tc=0
    for s in STYLES:
        rg,rt,coll,stp = agg[s]
        print(f"  {s:9s}{rg*100:8.0f}{rt*100:8.0f}{coll*100:7.0f}{stp:7.0f}")
        tg+=rg; tr+=rt; tc+=coll
    n=len(STYLES)
    print(f"  {'ALL':9s}{tg/n*100:8.0f}{tr/n*100:8.0f}{tc/n*100:7.0f}")""")

md("## 5. Comparison chart (round-trip % per style)")
code("""fig, ax = plt.subplots(figsize=(11,4.5))
x = np.arange(len(STYLES)); w = 0.8/max(len(rows),1)
colors = {"PPO":"#1f77b4", "RecurrentPPO":"#d6610a"}
for i,(name,agg) in enumerate(rows.items()):
    ax.bar(x+(i-(len(rows)-1)/2)*w, [agg[s][1]*100 for s in STYLES], w,
           label=name, color=colors.get(name,None), alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(STYLES); ax.set_ylim(0,105)
ax.set_ylabel("round-trip success %"); ax.set_title("Round-trip success by corridor style")
ax.legend(); ax.grid(axis="y", alpha=0.3); plt.tight_layout(); plt.show()""")

md("## 6. Top-down path grid (best policy)")
code("""from render2d import plot_paths

class WrapFF:   # feed-forward wrapper with auto-resetting frame stack
    def __init__(self, pol): self.p=pol; self.s=None; self.d=None; self.n=0
    def predict(self, obs, deterministic=True):
        o=np.asarray(obs,np.float32); pa=o[-3:-1]
        fresh=(self.s is None or self.d!=o.shape[0] or (abs(pa[0])<1e-6 and abs(pa[1])<1e-6 and self.n>3))
        if fresh: self.d=o.shape[0]; self.s=np.tile(o,self.p["frame_stack"]); self.n=0
        else: self.s=np.concatenate([self.s[self.d:],o])
        self.n+=1; a,_=self.p["model"].predict(self.p["norm"](self.s),deterministic=True); return a,None

class WrapLSTM:
    def __init__(self,pol): self.p=pol; self.state=None; self.n=0
    def predict(self, obs, deterministic=True):
        o=np.asarray(obs,np.float32); pa=o[-3:-1]
        if self.state is None or (abs(pa[0])<1e-6 and abs(pa[1])<1e-6 and self.n>3):
            self.state=None; self.n=0; es=np.ones(1,bool)
        else: es=np.zeros(1,bool)
        self.n+=1
        a,self.state=self.p["model"].predict(self.p["norm"](o)[None],state=self.state,
                                             episode_start=es,deterministic=True)
        return a[0],None

best = "RecurrentPPO" if "RecurrentPPO" in policies else "PPO"
pol = policies[best]
wrap = WrapLSTM(pol) if pol["recurrent"] else WrapFF(pol)
env = Nav2DEnv(config=ENV_CFG)
summ = plot_paths(wrap, env, list(STYLES), seeds=[7000+i for i in range(len(STYLES))],
                  savepath=os.path.join(RES,"paths_grid_nb.png"))
print(f"path grid for {best}:")
display(Image(filename=os.path.join(RES,"paths_grid_nb.png")))""")

md("""## 7. GIFs — robot driving each corridor type

Pre-recorded by `eval_ppo.py --gifs` (PPO) and the recurrent eval. Each GIF runs
until the robot finishes its round trip. If a GIF is missing, run the matching
eval script.""")
code("""gif_dir = os.path.join(RES, "gifs")
shown = 0
for style in STYLES:
    # prefer recurrent gif if present, else PPO gif
    for prefix in (["rec_", "nav2d_"] if best=="RecurrentPPO" else ["nav2d_", "rec_"]):
        path = os.path.join(gif_dir, f"{prefix}{style}.gif")
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            display(Markdown(f"**{style}**  (`{os.path.basename(path)}`)"))
            display(Image(filename=path)); shown += 1; break
print(f"shown {shown} GIFs from {gif_dir}")""")

md("""---
## 7b. HARD multi-corridor maps (junctions / grid / loop)

A separate policy `ppo_nav2d_hard` is trained on a **mixed pool** that adds
multi-corridor layouts: **T-junction, cross, double-T, grid, branch, loop**. Here
the robot must pick the right branch at each junction — using sensors only — then
do the round trip. Below: eval table on unseen hard maps + GIFs.""")
code("""hard_zip = os.path.join(RUNS, "ppo_nav2d_hard.zip")
HARD_STYLES = ("T_junction", "cross", "grid", "branch", "loop", "double_T")
if os.path.isfile(hard_zip):
    from world_hard import HARD_STYLES as _HS
    HARD_STYLES = _HS
    hvn = load_vecnorm(os.path.join(RUNS, "ppo_nav2d_hard_vecnorm.pkl"), FRAME_STACK)
    hard_pol = dict(model=PPO.load(hard_zip), norm=normalizer(hvn),
                    recurrent=False, frame_stack=FRAME_STACK)
    HCFG = dict(n_lidar=24, lidar_range=5.0, max_steps=1000, round_trip=True)
    def run_hard(seed, style):
        env = Nav2DEnv(config=HCFG); o, info = env.reset(seed=seed, options={"style": style})
        d = o.shape[0]; stack = np.tile(o, FRAME_STACK); done = False; n = 0; rg = rt = False
        while not done and n < 1000:
            a, _ = hard_pol["model"].predict(hard_pol["norm"](stack), deterministic=True)
            o, r, term, trunc, info = env.step(a); stack = np.concatenate([stack[d:], o]); n += 1
            rg = rg or info["reached_goal"]; rt = rt or info["round_trip"]; done = term or trunc
        return rg, rt
    EPH = 20
    print(f"=== HARD maps ({EPH} unseen/style) ===")
    print(f"  {'style':11s}{'reach%':>8s}{'round%':>8s}")
    tg = tr = 0
    for s in HARD_STYLES:
        rg = rt = 0
        for sd in range(EPH):
            a, b = run_hard(9000 + sd, s); rg += a; rt += b
        print(f"  {s:11s}{rg/EPH*100:8.0f}{rt/EPH*100:8.0f}"); tg += rg; tr += rt
    n = len(HARD_STYLES) * EPH
    print(f"  {'ALL':11s}{tg/n*100:8.0f}{tr/n*100:8.0f}")
else:
    print("no ppo_nav2d_hard model yet -> run: NAV2D_WORLD_KIND=mixed python train_ppo.py --save runs/ppo_nav2d_hard")""")
code("""# GIFs of the hard-map policy navigating each multi-corridor layout
hgif_dir = os.path.join(RES, "gifs"); shown = 0
for style in HARD_STYLES:
    path = os.path.join(hgif_dir, f"hard_{style}.gif")
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        display(Markdown(f"**{style}**")); display(Image(filename=path)); shown += 1
print(f"shown {shown} hard-map GIFs")""")

md("""---
## 9. Multi-stop delivery on FIXED maps (Hybrid: Global Planner + RL/LiDAR)

Now the robot does **real delivery**: hand-designed apartment maps with a wide
main corridor, many side niches (incl. a **curved arc niche** and **narrow hard
niches**), 10-12 fixed delivery points and a dock. You pick **1-3 points**; a
**TSP planner** computes the optimal visiting order using true path distances
(tiện đường), the robot delivers **all** of them, then returns to the **dock**
(battery-friendly single trip). Navigation is **hybrid**: a global A* planner
lays the route; the trained PPO sensor-policy follows it locally with LiDAR.""")
code("""from fixed_maps import build_map, map_names
from hybrid_runner import HybridDeliveryRunner
from render_delivery import plot_plan, record_delivery_gif
print("available maps:", map_names())

MAP_NAME = "apartment_a"          # "apartment_a" | "apartment_b"
runner = HybridDeliveryRunner(map_name=MAP_NAME)
print("delivery points:", sorted(runner.map["points"]))
print("dock at:", tuple(round(v,1) for v in runner.map["dock"]))""")

md("""### 9a. Pick the delivery points → optimal route (static plan)

Set `PICK` to any 1-3 point ids from the list above. The planner orders them to
minimise total distance and shows the route + battery estimate.""")
code("""PICK = [1, 5, 8]                  # <- choose 1-3 delivery point ids
plan = runner.dp.optimize(PICK)
print("chosen        :", PICK)
print("optimal order :", "DOCK -> " + " -> ".join(f"#{p}" for p in plan["order"]) + " -> DOCK")
print(f"total distance: {plan['total_dist']:.1f} m")
print(f"battery used  : ~{plan['battery_pct']:.0f} %")
for leg in plan["legs"]:
    print(f"   {leg['from']:>4} -> {str(leg['to']):>4} : {leg['dist']:.1f} m")
fig_dir = os.path.join(RES, "figs")
plot_plan(runner, plan, PICK, os.path.join(fig_dir, "delivery_plan.png"),
          title=f"{MAP_NAME} — deliver {PICK}")
display(Image(filename=os.path.join(fig_dir, "delivery_plan.png")))""")

md("### 9b. Run it → GIF of the robot delivering all stops then returning to dock")
code("""gif = os.path.join(RES, "gifs", f"delivery_{MAP_NAME}_{''.join(map(str,PICK))}.gif")
res = record_delivery_gif(runner, PICK, gif, max_frames=150)
print(f"success={res['success']}  delivered={res.get('delivered')}  "
      f"returned_to_dock={res.get('returned_dock')}")
print(f"sim_time={res.get('sim_time_s',0):.1f}s  distance={res['total_dist']:.1f}m  "
      f"battery~{res['battery_pct']:.0f}%")
display(Image(filename=gif))""")

md("""### 9c. A few more combinations (plans only — fast)""")
code("""for pick in ([2, 8], [0, 3, 9], [7, 4, 6]):
    pick = [p for p in pick if p in runner.map["points"]]
    pl = runner.dp.optimize(pick)
    print(f"  pick {pick}: DOCK->{pl['order']}->DOCK  {pl['total_dist']:.1f} m  ~{pl['battery_pct']:.0f}% batt")""")

md("""---
## 10. Multi-stop delivery — LSTM, Train == Deploy (stable driving)

The earlier delivery controller weaved and often failed to return. This is the
rebuilt version following the **Train == Deploy** design: one env
(`multistop_env`) runs the whole trip — the **Global Planner (A*+TSP)** picks &
orders the stops in `reset()`, and on reaching a stop the env does a **Continuous
Goal Transition** (no reset; load the next leg) with a **grace period** so the
robot can turn around in a niche without being punished. The brain is a
**RecurrentPPO (LSTM)** (memory to back out of dead ends), the robot may **reverse**,
and light bumps are forgiven (**collision-grace**). Evaluation runs the policy
**directly in this env** — no external runner, so no train/deploy mismatch.""")
code("""from eval_multistop import load as load_ms, CFG as MS_CFG
from multistop_env import MultiStopEnv
ms_zip = os.path.join(RUNS, "ms_lstm.zip")
if os.path.isfile(ms_zip):
    ms_model, ms_norm = load_ms(os.path.join(RUNS, "ms_lstm"),
                                os.path.join(RUNS, "ms_lstm_vecnorm.pkl"))
    from eval_multistop import run_episode
    msenv = MultiStopEnv(config=MS_CFG)
    EP = 20
    print(f"=== multi-stop LSTM ({EP} trips/map) ===")
    print(f"  {'map':12s}{'full-trip%':>11s}{'stops':>9s}{'coll%':>7s}")
    for name in msenv.map_names:
        dock=sd=tot=coll=0
        for i in range(EP):
            r = run_episode(ms_model, ms_norm, msenv, seed=9000+i, options={"map":name})
            dock+=int(r["dock"]); sd+=r["stops"]; tot+=r["stops_total"]; coll+=int(r["coll"])
        print(f"  {name:12s}{dock/EP*100:11.0f}{f'{sd}/{tot}':>9s}{coll/EP*100:7.0f}")
else:
    print("no ms_lstm model yet -> run: python nav2d/train_multistop.py --timesteps 2000000 --subproc")""")
md("### 10a. GIFs — full multi-stop trip (deliver all stops, return to dock)")
code("""for name in (MultiStopEnv(config=MS_CFG).map_names if os.path.isfile(ms_zip) else []):
    g = os.path.join(RES, "gifs", f"multistop_{name}.gif")
    if os.path.isfile(g):
        display(Markdown(f"**{name}**")); display(Image(filename=g))""")

md("""---
## 11. Generalization — train on RANDOM maps, test ZERO-SHOT on unseen maps

The ultimate goal: a robot that drives any building it's never seen. We train the
LSTM on an **endless stream of procedurally-generated maps** (a new random
multi-corridor layout every episode) plus **domain randomization** (LiDAR noise +
wheel-gain/slip jitter), so it can't memorise — it must learn the *rule* (follow
the planned route, dodge walls by LiDAR). Then we test **zero-shot** on the
hand-made maps a/b/c, which were **never seen during this training**.""")
code("""import numpy as np
from multistop_env import MultiStopEnv
from eval_multistop import run_episode as _run_ep
MSC = dict(n_lidar=24, lidar_range=5.0, lookahead=1.6, max_steps=2000,
           grace_steps=18, collision_grace=25, reverse_frac=0.4)
proc_zip = os.path.join(RUNS, "ms_proc.zip")
if os.path.isfile(proc_zip):
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    pm = RecurrentPPO.load(os.path.join(RUNS, "ms_proc"))
    pv = VecNormalize.load(os.path.join(RUNS, "ms_proc_vecnorm.pkl"),
                           DummyVecEnv([lambda: MultiStopEnv(config=dict(MSC, procedural=True))]))
    pmean = pv.obs_rms.mean.astype(np.float32); pvar = pv.obs_rms.var.astype(np.float32)
    pnorm = lambda o: np.clip((o-pmean)/np.sqrt(pvar+pv.epsilon), -pv.clip_obs, pv.clip_obs).astype(np.float32)

    def eval_env(env, name, EP=20, opts=None):
        dock=sd=tot=coll=0
        for i in range(EP):
            r=_run_ep(pm, pnorm, env, seed=9000+i)  # run_episode resets internally
            # run_episode uses its own reset; for map control we re-run inline:
        # simpler inline eval to honour the map option:
        dock=sd=tot=coll=0
        for i in range(EP):
            o,info=env.reset(seed=9000+i, options=opts); st=None; es=np.ones(1,bool); done=False; s=0; d=False; c=False
            while not done:
                a,st=pm.predict(pnorm(o)[None], state=st, episode_start=es, deterministic=True)
                es=np.zeros(1,bool); o,r,t,tr,info=env.step(a[0]); s=max(s,info["stops_done"]); d=d or info["arrived_dock"]; c=c or info["collision"]; done=t or tr
            dock+=int(d); sd+=s; tot+=env.stops_total; coll+=int(c)
        print(f"  {name:30s} full-trip {dock/EP*100:4.0f}%  stops {sd}/{tot}  collision {coll/EP*100:4.0f}%")

    print("=== IN-DISTRIBUTION (procedural random maps + noise) ===")
    eval_env(MultiStopEnv(config=dict(MSC, procedural=True, domain_random=True)), "procedural+noise")
    print("=== ZERO-SHOT (hand-made maps NEVER trained on) ===")
    he = MultiStopEnv(config=dict(MSC, procedural=False, domain_random=False,
                                  maps=["apartment_a","apartment_b","apartment_c"]))
    for nm in ["apartment_a","apartment_b","apartment_c"]:
        eval_env(he, "zero-shot "+nm, opts={"map":nm})
else:
    print("no ms_proc model yet -> run: NAV2D_PROCEDURAL=1 python nav2d/train_multistop.py --timesteps 4000000 --subproc --save runs/ms_proc")""")
md("### 11a. Zero-shot GIFs — the robot delivering on maps it never trained on")
code("""for nm in ["apartment_a","apartment_b","apartment_c"]:
    g = os.path.join(RES, "gifs", f"zeroshot_{nm}.gif")
    if os.path.isfile(g):
        display(Markdown(f"**zero-shot {nm}**")); display(Image(filename=g))""")

md("""---
## 12. 🎮 Test on apartment_c — pick your own delivery points

Interactive test on the hard map **apartment_c** (vertical niches + a nested
**arc corridor** with a **sub-niche** at its middle) using the zero-shot model
`ms_proc` (trained only on random procedural maps — it never saw apartment_c).
Set `C_PICK` to any 1-3 point ids from the table, run the cell, and it plans the
optimal order, runs the robot, prints the result, and shows the trip GIF.""")
code("""from fixed_maps import build_map as _bm
from multistop_env import MultiStopEnv
from render_multistop import record_trip_gif
import numpy as np

_mc = _bm("apartment_c")
print("apartment_c delivery points (id : x, y):")
for pid, xy in sorted(_mc["points"].items()):
    tag = "  <- arc sub-niche (hard)" if pid == 5 else ("  <- arc arm" if pid in (6,7) else "")
    print(f"  {pid}: ({xy[0]:5.1f},{xy[1]:5.1f}){tag}")
print("dock:", tuple(round(v,1) for v in _mc["dock"]))""")
code("""# ====== CHOOSE HERE (1-3 ids from the table above) ======
C_PICK = [5, 6, 1]      # e.g. include 5 to make it visit the arc sub-niche
C_SEED = 9000
# ========================================================
MSC = dict(n_lidar=24, lidar_range=5.0, lookahead=1.6, max_steps=2000,
           grace_steps=18, collision_grace=25, reverse_frac=0.4,
           procedural=False, domain_random=False, maps=["apartment_c"])
cenv = MultiStopEnv(config=MSC)
gif = os.path.join(RES, "gifs", f"apt_c_pick_{''.join(map(str,C_PICK))}.gif")
# 'ms_proc' model + its normalizer (pm/pnorm) were loaded in section 11
res = record_trip_gif(pm, pnorm, cenv, gif, seed=C_SEED,
                      options={"map": "apartment_c", "points": list(C_PICK)}, max_frames=180)
print(f"picked {C_PICK} -> visit order DOCK -> "
      + " -> ".join(f"#{p}" for p in res['order']) + " -> DOCK")
print(f"returned to dock: {res['dock']}   |   steps: {res['steps']}")
print("(zero-shot model: apartment_c was never in its training set)")
display(Image(filename=gif))""")

md("""---
## 13. Curved corridors — final generalization model (`ms_mixed`)

The procedural generator was extended with **curved corridors** (arc / S-bend /
zigzag / hairpin-U) and **dead-end + nested niches**, the reward got a **dynamic
cross-track grace** (relax centre-line penalty in sharp bends so the robot may
cut the inside) + **doubled smoothness penalty**, and the model was trained
**fresh** on a **50/50 grid+curve curriculum** (LSTM, n_steps 1024) with
**auto early-stopping** (it stopped itself at 1.88M — no blind training).

Result vs the previous generalization model (`ms_proc_4M`, no curves in its
training) on held-out **curve** test maps + the apartment maps (regression check).""")
code("""import numpy as np, math
from multistop_env import MultiStopEnv
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
MSC = dict(n_lidar=24, lidar_range=5.0, lookahead=1.6, max_steps=2000,
           grace_steps=18, collision_grace=25, reverse_frac=0.4)
def _load(p):
    m = RecurrentPPO.load(os.path.join(RUNS, p))
    v = VecNormalize.load(os.path.join(RUNS, p+"_vecnorm.pkl"),
                          DummyVecEnv([lambda: MultiStopEnv(config=dict(MSC, procedural=True))]))
    me=v.obs_rms.mean.astype(np.float32); va=v.obs_rms.var.astype(np.float32)
    return m, (lambda o: np.clip((o-me)/np.sqrt(va+v.epsilon), -v.clip_obs, v.clip_obs).astype(np.float32))
def _smooth(tr):
    t=np.array(tr)
    if len(t)<3: return 0.0
    h=np.arctan2(np.diff(t[:,1]),np.diff(t[:,0])); dh=np.abs((np.diff(h)+math.pi)%(2*math.pi)-math.pi)
    return float(np.degrees(np.mean(dh)))
def _eval(m,norm,maps,EP=12):
    env=MultiStopEnv(config=dict(MSC,procedural=False,domain_random=False,maps=maps))
    for nm in maps:
        dock=s=tot=coll=0
        for i in range(EP):
            o,info=env.reset(seed=700+i,options={"map":nm}); st=None;es=np.ones(1,bool);done=False;sd=0;d=False;c=False
            while not done:
                a,st=m.predict(norm(o)[None],state=st,episode_start=es,deterministic=True)
                es=np.zeros(1,bool); o,r,t,tr,inf=env.step(a[0]); sd=max(sd,inf["stops_done"]); d=d or inf["arrived_dock"]; c=c or inf["collision"]; done=t or tr
            dock+=int(d); s+=sd; tot+=env.stops_total; coll+=int(c)
        print(f"  {nm:14s} dock {dock/EP*100:4.0f}%  stops {s}/{tot}  collision {coll/EP*100:3.0f}%")
cv=["test_c_curve","test_s_curve","test_u_turn"]; ap=["apartment_a","apartment_b","apartment_c"]
if os.path.isfile(os.path.join(RUNS,"ms_mixed.zip")):
    print("=== ms_mixed (curves + mixed curriculum) — held-out CURVE maps ===")
    m,n=_load("ms_mixed"); _eval(m,n,cv)
    print("=== ms_mixed — apartment maps (regression check) ===")
    _eval(m,n,ap)
    print("=== ms_proc_4M (no curves in training) — CURVE maps, for comparison ===")
    m0,n0=_load("ms_proc_4M"); _eval(m0,n0,cv)
else:
    print("no ms_mixed model -> NAV2D_PROCEDURAL=1 python nav2d/train_multistop.py --early-stop --save runs/ms_mixed")""")
md("### 13a. GIFs — `ms_mixed` on curved maps (C / S / U-turn) + apartment_c")
code("""for nm in ["test_c_curve","test_s_curve","test_u_turn","apartment_c"]:
    g=os.path.join(RES,"gifs",f"curve_{nm}.gif")
    if os.path.isfile(g):
        display(Markdown(f"**{nm}**")); display(Image(filename=g))""")

md("""---
## 14. Stage 2 — SLAM (blind mapping) → TSP/A* → delivery

The robot is dropped on a map with **no prior knowledge**. Phase 1: it runs
**frontier exploration** with a 2D LiDAR, integrating each scan into an
**occupancy grid** (log-odds) until it has mapped the reachable area. Phase 2: the
delivery-point coordinates are loaded, the **TSP** picks the visit order and
**A\\*** routes on the *discovered* grid. Phase 3: the trained LSTM policy drives
the route, delivers every stop and returns to the dock — all on a map it built
itself.""")
code("""from slam_delivery import run_slam_delivery
print("=== SLAM -> TSP -> deliver on maps the robot maps from scratch ===")
slam_runs = {}
for name, pts in [("apartment_c", [1, 2, 5]), ("test_s_curve", [0, 1, 2]),
                  ("test_u_turn", [0, 1, 2])]:
    r = run_slam_delivery(name, pts, seed=0)
    slam_runs[name] = r
    print(f"  {name:13s} explore_steps={r['explore_steps']:4d}  "
          f"points_reachable={r['points_reachable']}  order={r['order']}  "
          f"deliver {r['stops_done']}/{r['stops_total']-1}  "
          f"dock={r['returned_dock']}  collision={r['collision']}")""")
md("### 14a. GIFs — Phase 1 blind mapping (grid fills in) then Phase 2 delivery")
code("""for name in ["slam_apartment_c", "slam_test_u_turn"]:
    g = os.path.join(RES, "gifs", f"{name}.gif")
    if os.path.isfile(g):
        display(Markdown(f"**{name}**")); display(Image(filename=g))""")

md("""---
## 15. Stage 3 — dynamic obstacles (moving people) via a LiDAR safety shield

People move through the corridors. Instead of retraining one network to do
curves + multi-stop + dynamic dodging at once (which plateaued), we keep the
proven `ms_mixed` policy as the motion core and add a **reactive safety shield**
that runs on top: it scans a **narrow forward LiDAR cone**, subtracts the known
wall returns to isolate **moving obstacles**, and overrides the command — brake /
reverse if a person is very close, steer to the freer side in the slow band, and
hand control back to the RL policy once clear.""")
code("""from ped_shield import PedShield
from multistop_env import MultiStopEnv
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np
PCFG = dict(n_lidar=24, lidar_range=5.0, lookahead=1.6, max_steps=2500, grace_steps=18,
            collision_grace=25, reverse_frac=0.4, procedural=False, domain_random=False,
            n_ped=3, lidar_stack=1)
if os.path.isfile(os.path.join(RUNS, "ms_mixed.zip")):
    pm = RecurrentPPO.load(os.path.join(RUNS, "ms_mixed"))
    pv = VecNormalize.load(os.path.join(RUNS, "ms_mixed_vecnorm.pkl"),
                           DummyVecEnv([lambda: MultiStopEnv(config=dict(PCFG, procedural=True))]))
    pmean=pv.obs_rms.mean.astype(np.float32); pvar=pv.obs_rms.var.astype(np.float32)
    pnorm=lambda o: np.clip((o-pmean)/np.sqrt(pvar+pv.epsilon), -pv.clip_obs, pv.clip_obs).astype(np.float32)
    shield = PedShield()
    def wallscan(env):
        ang = env.heading + env.lidar_angles
        return env.world.raycast_batch(tuple(env.pos), ang, env.lidar_range) / env.lidar_range
    def run(use_shield, EP=12):
        env = MultiStopEnv(config=dict(PCFG, maps=["apartment_a","apartment_b","apartment_c"]))
        dock=stops=tot=coll=0
        for i in range(EP):
            o,info=env.reset(seed=800+i); st=None; es=np.ones(1,bool); done=False; s=0; d=False; c=False
            while not done:
                a,st=pm.predict(pnorm(o)[None], state=st, episode_start=es, deterministic=True); a=a[0]
                if use_shield:
                    a,_=shield.filter_lidar(a, o[:env.n_lidar], env.lidar_angles,
                                            env.lidar_range, wall_lidar=wallscan(env))
                es=np.zeros(1,bool); o,r,t,tr,inf=env.step(a)
                s=max(s,inf["stops_done"]); d=d or inf["arrived_dock"]; c=c or inf["collision"]; done=t or tr
            dock+=int(d); stops+=s; tot+=env.stops_total; coll+=int(c)
        print(f"  [{'WITH LiDAR shield' if use_shield else 'no shield':17s}] dock {dock/EP*100:4.0f}%  "
              f"stops {stops}/{tot}  pedestrian-collision {coll/EP*100:3.0f}%")
    print("=== ms_mixed + 3 moving people (apartment a/b/c) ===")
    run(False); run(True)
else:
    print("no ms_mixed model")""")
md("### 15a. GIF — a successful delivery run dodging moving people")
code("""g = os.path.join(RES, "gifs", "dodge_pedestrians.gif")
if os.path.isfile(g):
    display(Image(filename=g))
else:
    print("run render_dodge.py to generate the dodge GIF")""")

md("""## 8. Conclusion

- Both policies navigate **unseen** corridors of every shape using **sensors only**
  — no map, no planner — and complete the **round trip** (reach goal, return to
  start). Reach-goal is ~100% for both.
- **Measured (20 unseen maps/style):** feed-forward **PPO + frame-stack** ≈ 98%
  round-trip overall; **RecurrentPPO (LSTM)** ≈ 94% at 800k steps. The LSTM
  improves the hardest **U-turn** case (its memory helps the 180° turn) but, at
  this budget, trails on `niches`; with more steps it typically catches up. So
  for this map set **PPO is the better-rounded default**, LSTM is the path to
  longer-memory tasks (mazes, junctions).
- Next: harder maps (junctions, multi-door rooms, dynamic obstacles), then port
  the behaviour into the PyBullet `delivery_rl` env.
- **Hard multi-corridor result (hard-only specialist, 3M steps):** after making
  it a **specialist** (train on the hard pool only) and fixing the tight-corridor
  behaviour — a real **clearance penalty** (the old `w_clear` was a no-op bug) plus
  a **light recoverable bump** instead of an instant-terminate collision — the
  robot now does the round trip on **all six layouts ≈ 99%**, with the dead-end
  **branch** jumping from ~25% to **~92%**. The simple single-corridor policy
  `ppo_nav2d` remains the specialist for those maps (~98%).""")

nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3",
      "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for t, s in cells:
    c = {"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True)}
    if t == "code":
        c["execution_count"] = None; c["outputs"] = []
    nb["cells"].append(c)
with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote", NB, "| cells:", len(cells))
