"""So sánh ĐẦY ĐỦ ms_guided (RL học pure-pursuit) vs Pure-Pursuit (thầy).
Set A: 100 episode trên phân phối 1000 map train (held-out theo seed).
Set B: các map CŨ đã test (apartment a/b/c, 3 cong, apartment_complex).
Báo tỉ lệ %: tới đích, va chạm, bám tim đường, đánh võng, thời gian, độ giống PP.
"""
import sys, math
sys.path.insert(0, ".")
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from track_env import TrackEnv, W_MAX
from fixed_maps import build_map
import nav_demo as nd

def smoothness(trail):
    tr = np.array(trail); d = np.diff(tr, axis=0); n = np.linalg.norm(d, axis=1); d = d[n > 1e-4]
    if len(d) < 2: return 0.0
    ang = np.arctan2(d[:, 1], d[:, 0])
    return float(np.degrees(np.abs((np.diff(ang) + math.pi) % (2*math.pi) - math.pi).mean()))

model = PPO.load("runs/ms_guided")
_vec = VecNormalize.load("runs/ms_guided_vecnorm.pkl", DummyVecEnv([lambda: TrackEnv()]))
mean = _vec.obs_rms.mean.astype(np.float32); var = _vec.obs_rms.var.astype(np.float32)
norm = lambda o: np.clip((o-mean)/np.sqrt(var+_vec.epsilon), -_vec.clip_obs, _vec.clip_obs).astype(np.float32)

def episode(env, seed, mode):
    o, i = env.reset(seed=seed); done = False
    trail = [tuple(env.pos)]; eys=[]; gaps=[]; arrived=False; coll=False; steps=0
    while not done:
        wpp = env._omega_pp()
        if mode == "RL":
            a, _ = model.predict(norm(o)[None], deterministic=True); a = a[0]
        else:
            a = np.array([1.0, wpp / W_MAX], np.float32)
        o, r, t, tr, inf = env.step(a)
        eys.append(abs(env._nearest_seg()[1])); gaps.append(abs(inf["omega_rl"]-inf["omega_pp"]))
        trail.append(tuple(env.pos)); steps += 1
        arrived = inf.get("arrived", False); coll = coll or inf.get("collision", False); done = t or tr
    return dict(arr=arrived, coll=coll, ey=float(np.mean(eys)), smooth=smoothness(trail),
                steps=steps, gap=float(np.mean(gaps)))

def summarize(name, env, seeds):
    agg = {"Pure-Pursuit": [], "RL": []}
    for sd in seeds:
        for mode in agg:
            agg[mode].append(episode(env, sd, mode))
    print(f"\n=== {name} ({len(seeds)} lượt) ===")
    print(f"{'Bộ lái':16s}{'Tới đích':>10s}{'Va chạm':>9s}{'e_y(m)':>8s}{'Đánh võng°':>12s}{'Bước':>7s}{'Lệch PP':>9s}")
    out = {}
    for mode, rs in agg.items():
        arr=np.mean([r['arr'] for r in rs])*100; co=np.mean([r['coll'] for r in rs])*100
        ey=np.mean([r['ey'] for r in rs]); sm=np.mean([r['smooth'] for r in rs])
        st=np.mean([r['steps'] for r in rs]); gp=np.mean([r['gap'] for r in rs])
        nm = "RL ms_guided" if mode=="RL" else mode
        print(f"{nm:16s}{arr:>9.0f}%{co:>8.0f}%{ey:>8.3f}{sm:>12.2f}{st:>7.0f}{gp:>9.3f}")
        out[mode]=dict(arr=arr,co=co,ey=ey,sm=sm,st=st)
    return out

# Set A: phân phối train (held-out seeds)
A = summarize("SET A — 1000-map distribution", TrackEnv(config=dict(domain_random=False)),
              list(range(2000, 2100)))

# Set B: map cũ
fixed = []
for nm in ["apartment_a","apartment_b","apartment_c","test_c_curve","test_s_curve","test_u_turn"]:
    m = build_map(nm); fixed.append({"grid":m["grid"],"cell":m["cell"],"origin":m["origin"],"dock":m["dock"],"points":m["points"]})
sm = nd.load_saved_map(); fixed.append({"grid":sm["grid"],"cell":sm["cell"],"origin":sm["origin"],"dock":sm["dock"],"points":sm["points"]})
B = summarize("SET B — map CŨ (apt a/b/c + 3 cong + apartment_complex)",
              TrackEnv(config=dict(domain_random=False, maps=fixed)), list(range(50)))

# tỉ lệ "RL đạt bao nhiêu % so với pure-pursuit"
print("\n=== TỈ LỆ RL so với Pure-Pursuit ===")
for nm, d in [("Set A", A), ("Set B", B)]:
    arr_ratio = d["RL"]["arr"]/max(d["Pure-Pursuit"]["arr"],1e-9)*100
    sm_ratio  = d["Pure-Pursuit"]["sm"]/max(d["RL"]["sm"],1e-9)*100   # mượt: PP/RL (100%=bằng)
    print(f"{nm}: tới đích RL = {arr_ratio:.0f}% của PP | độ mượt RL = {sm_ratio:.0f}% của PP "
          f"(RL {d['RL']['sm']:.2f}° vs PP {d['Pure-Pursuit']['sm']:.2f}°)")
print("DONE")
