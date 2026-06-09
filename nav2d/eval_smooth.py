"""So sánh Pure-Pursuit vs ms_guided (K=0.5) vs ms_guided_smooth (K=1.0 + phạt Δω).
Mục tiêu: ms_guided_smooth có đánh võng <= pure-pursuit không."""
import sys, math
sys.path.insert(0, ".")
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from track_env import TrackEnv, W_MAX
from fixed_maps import build_map
import nav_demo as nd

def smoothness(trail):
    tr=np.array(trail); d=np.diff(tr,axis=0); n=np.linalg.norm(d,axis=1); d=d[n>1e-4]
    if len(d)<2: return 0.0
    ang=np.arctan2(d[:,1],d[:,0])
    return float(np.degrees(np.abs((np.diff(ang)+math.pi)%(2*math.pi)-math.pi).mean()))

def load(name):
    m=PPO.load(f"runs/{name}")
    v=VecNormalize.load(f"runs/{name}_vecnorm.pkl", DummyVecEnv([lambda: TrackEnv()]))
    mu=v.obs_rms.mean.astype(np.float32); va=v.obs_rms.var.astype(np.float32)
    return m, (lambda o: np.clip((o-mu)/np.sqrt(va+v.epsilon),-v.clip_obs,v.clip_obs).astype(np.float32))

MODELS={"ms_guided_smooth":load("ms_guided_smooth"), "ms_guided_smooth2":load("ms_guided_smooth2")}

def episode(env, seed, mode):
    o,i=env.reset(seed=seed); done=False; trail=[tuple(env.pos)]; eys=[]; arrived=False; coll=False; steps=0
    while not done:
        wpp=env._omega_pp()
        if mode=="Pure-Pursuit":
            a=np.array([1.0, wpp/W_MAX], np.float32)
        else:
            mdl,nrm=MODELS[mode]; a,_=mdl.predict(nrm(o)[None],deterministic=True); a=a[0]
        o,r,t,tr,inf=env.step(a)
        eys.append(abs(env._nearest_seg()[1])); trail.append(tuple(env.pos)); steps+=1
        arrived=inf.get("arrived",False); coll=coll or inf.get("collision",False); done=t or tr
    return dict(arr=arrived,coll=coll,ey=float(np.mean(eys)),smooth=smoothness(trail),steps=steps)

def summarize(name, env, seeds):
    print(f"\n=== {name} ({len(seeds)} lượt) ===")
    print(f"{'Bộ lái':20s}{'Tới đích':>9s}{'Va chạm':>8s}{'e_y(m)':>8s}{'Đánh võng°':>12s}{'Bước':>7s}")
    for mode in ["Pure-Pursuit","ms_guided_smooth","ms_guided_smooth2"]:
        rs=[episode(env,sd,mode) for sd in seeds]
        arr=np.mean([r['arr'] for r in rs])*100; co=np.mean([r['coll'] for r in rs])*100
        ey=np.mean([r['ey'] for r in rs]); sm=np.mean([r['smooth'] for r in rs]); st=np.mean([r['steps'] for r in rs])
        print(f"{mode:20s}{arr:>8.0f}%{co:>7.0f}%{ey:>8.3f}{sm:>12.2f}{st:>7.0f}")

summarize("SET A — 1000-map distribution", TrackEnv(config=dict(domain_random=False)), list(range(2000,2070)))
fixed=[]
for nm in ["apartment_a","apartment_b","apartment_c","test_c_curve","test_s_curve","test_u_turn"]:
    m=build_map(nm); fixed.append({"grid":m["grid"],"cell":m["cell"],"origin":m["origin"],"dock":m["dock"],"points":m["points"]})
sm=nd.load_saved_map(); fixed.append({"grid":sm["grid"],"cell":sm["cell"],"origin":sm["origin"],"dock":sm["dock"],"points":sm["points"]})
summarize("SET B — map cũ", TrackEnv(config=dict(domain_random=False, maps=fixed)), list(range(50)))
print("DONE")
