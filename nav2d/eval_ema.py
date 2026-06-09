"""Thử lọc EMA lên ω của RL (lúc chạy, không train lại) để đạt đánh võng <= PP.
omega_smoothed = alpha*omega_prev + (1-alpha)*omega_RL.
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
    tr=np.array(trail); d=np.diff(tr,axis=0); n=np.linalg.norm(d,axis=1); d=d[n>1e-4]
    if len(d)<2: return 0.0
    ang=np.arctan2(d[:,1],d[:,0])
    return float(np.degrees(np.abs((np.diff(ang)+math.pi)%(2*math.pi)-math.pi).mean()))

m=PPO.load("runs/ms_guided_smooth")
v=VecNormalize.load("runs/ms_guided_smooth_vecnorm.pkl", DummyVecEnv([lambda: TrackEnv()]))
mu=v.obs_rms.mean.astype(np.float32); va=v.obs_rms.var.astype(np.float32)
nrm=lambda o: np.clip((o-mu)/np.sqrt(va+v.epsilon),-v.clip_obs,v.clip_obs).astype(np.float32)

def episode(env, seed, mode, alpha=0.0):
    o,i=env.reset(seed=seed); done=False; trail=[tuple(env.pos)]; arrived=False; coll=False; steps=0
    w_ema=0.0
    while not done:
        wpp=env._omega_pp()
        if mode=="Pure-Pursuit":
            a=np.array([1.0, wpp/W_MAX], np.float32)
        else:
            act,_=m.predict(nrm(o)[None],deterministic=True); act=act[0]
            w_rl=float(act[1])*W_MAX
            w_ema=alpha*w_ema+(1-alpha)*w_rl          # lọc EMA
            a=np.array([act[0], np.clip(w_ema/W_MAX,-1,1)], np.float32)
        o,r,t,tr,inf=env.step(a)
        trail.append(tuple(env.pos)); steps+=1
        arrived=inf.get("arrived",False); coll=coll or inf.get("collision",False); done=t or tr
    return arrived, coll, smoothness(trail), steps

def run(name, env, seeds):
    print(f"\n=== {name} ({len(seeds)} lượt) ===")
    print(f"{'Bộ lái':28s}{'Tới đích':>9s}{'Va chạm':>8s}{'Đánh võng°':>12s}{'Bước':>7s}")
    configs=[("Pure-Pursuit",0.0),("RL smooth (no filter)",0.0),
             ("RL + EMA α=0.6",0.6),("RL + EMA α=0.8",0.8)]
    for label,al in configs:
        mode="Pure-Pursuit" if label.startswith("Pure") else "RL"
        rs=[episode(env,sd,mode,al) for sd in seeds]
        arr=np.mean([r[0] for r in rs])*100; co=np.mean([r[1] for r in rs])*100
        sm=np.mean([r[2] for r in rs]); st=np.mean([r[3] for r in rs])
        print(f"{label:28s}{arr:>8.0f}%{co:>7.0f}%{sm:>12.2f}{st:>7.0f}")

run("SET A — 1000-map", TrackEnv(config=dict(domain_random=False)), list(range(2000,2060)))
fixed=[]
for nm in ["apartment_a","apartment_b","apartment_c","test_c_curve","test_s_curve","test_u_turn"]:
    mm=build_map(nm); fixed.append({"grid":mm["grid"],"cell":mm["cell"],"origin":mm["origin"],"dock":mm["dock"],"points":mm["points"]})
sm=nd.load_saved_map(); fixed.append({"grid":sm["grid"],"cell":sm["cell"],"origin":sm["origin"],"dock":sm["dock"],"points":sm["points"]})
run("SET B — map cũ", TrackEnv(config=dict(domain_random=False, maps=fixed)), list(range(40)))
print("DONE")
