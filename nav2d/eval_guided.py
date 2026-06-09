"""So sánh ms_guided (RL học pure-pursuit) vs Pure-Pursuit (thầy) trong track_env.
Đo: tới đích?, bám tim đường |e_y|, đánh võng (deg/step), số bước, lệch so với PP.
Cùng seed -> cùng map/tuyến -> công bằng.
"""
import sys, math
sys.path.insert(0, ".")
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from track_env import TrackEnv, W_MAX

def smoothness(trail):
    tr = np.array(trail); d = np.diff(tr, axis=0); n = np.linalg.norm(d, axis=1); d = d[n > 1e-4]
    if len(d) < 2: return 0.0
    ang = np.arctan2(d[:, 1], d[:, 0])
    return float(np.degrees(np.abs((np.diff(ang) + math.pi) % (2*math.pi) - math.pi).mean()))

# load model + vecnorm
model = PPO.load("runs/ms_guided")
vec = VecNormalize.load("runs/ms_guided_vecnorm.pkl", DummyVecEnv([lambda: TrackEnv()]))
mean = vec.obs_rms.mean.astype(np.float32); var = vec.obs_rms.var.astype(np.float32)
norm = lambda o: np.clip((o-mean)/np.sqrt(var+vec.epsilon), -vec.clip_obs, vec.clip_obs).astype(np.float32)

env = TrackEnv(config=dict(domain_random=False))

def episode(seed, mode):
    o, i = env.reset(seed=seed); done = False
    trail = [tuple(env.pos)]; eys = []; gaps = []; arrived = False; steps = 0
    while not done:
        wpp = env._omega_pp()
        if mode == "RL ms_guided":
            a, _ = model.predict(norm(o)[None], deterministic=True); a = a[0]
        else:  # Pure-pursuit (thầy)
            a = np.array([1.0, wpp / W_MAX], np.float32)
        o, r, t, tr, inf = env.step(a)
        eys.append(abs(env._nearest_seg()[1])); gaps.append(abs(inf["omega_rl"] - inf["omega_pp"]))
        trail.append(tuple(env.pos)); steps += 1; arrived = inf.get("arrived", False); done = t or tr
    return dict(arr=arrived, ey=float(np.mean(eys)), smooth=smoothness(trail),
                steps=steps, gap=float(np.mean(gaps)))

N = 40
res = {"Pure-Pursuit (thầy)": [], "RL ms_guided": []}
for sd in range(N):
    for mode in res:
        res[mode].append(episode(1000 + sd, mode))

print(f"{'Bộ lái':22s}{'Tới đích':>10s}{'e_y(m)':>9s}{'Đánh võng°':>12s}{'Bước':>7s}{'Lệch PP':>9s}")
print("-"*70)
for mode, rs in res.items():
    arr = np.mean([r["arr"] for r in rs]) * 100
    ey = np.mean([r["ey"] for r in rs]); sm = np.mean([r["smooth"] for r in rs])
    st = np.mean([r["steps"] for r in rs]); gp = np.mean([r["gap"] for r in rs])
    print(f"{mode:22s}{arr:>9.0f}%{ey:>9.3f}{sm:>12.2f}{st:>7.0f}{gp:>9.3f}")
print("\n(e_y nhỏ = bám tim đường tốt; đánh võng nhỏ = mượt; Lệch PP nhỏ = giống thầy)")
