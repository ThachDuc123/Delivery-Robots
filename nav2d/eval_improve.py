"""So sánh đánh võng: Pure-Pursuit vs ms_guided_smooth vs ms_guided_caps (CAPS) vs
ms_guided_lipsnet (Lipschitz). Set A (1000-map) + Set B (map cũ)."""
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
    if "lipsnet" in name:
        # dựng lại policy có spectral_norm rồi nạp tham số
        from train_lipsnet import constrain_actor_lipschitz
        m=PPO("MlpPolicy", DummyVecEnv([lambda: TrackEnv()]), policy_kwargs=dict(net_arch=[256,256]))
        constrain_actor_lipschitz(m.policy)
        m.set_parameters(f"runs/{name}")
    else:
        m=PPO.load(f"runs/{name}")
    v=VecNormalize.load(f"runs/{name}_vecnorm.pkl", DummyVecEnv([lambda: TrackEnv()]))
    mu=v.obs_rms.mean.astype(np.float32); va=v.obs_rms.var.astype(np.float32)
    return m,(lambda o: np.clip((o-mu)/np.sqrt(va+v.epsilon),-v.clip_obs,v.clip_obs).astype(np.float32))

M={n:load(n) for n in ["ms_guided_smooth","ms_guided_caps","ms_guided_lipsnet"]}

def episode(env, seed, mode):
    o,i=env.reset(seed=seed); done=False; trail=[tuple(env.pos)]; eys=[]; arr=False; coll=False; st=0
    while not done:
        wpp=env._omega_pp()
        if mode=="Pure-Pursuit": a=np.array([1.0,wpp/W_MAX],np.float32)
        else:
            mdl,nrm=M[mode]; act,_=mdl.predict(nrm(o)[None],deterministic=True); a=act[0]
        o,r,t,tr,inf=env.step(a)
        eys.append(abs(env._nearest_seg()[1])); trail.append(tuple(env.pos)); st+=1
        arr=inf.get("arrived",False); coll=coll or inf.get("collision",False); done=t or tr
    return dict(arr=arr,coll=coll,ey=float(np.mean(eys)),smooth=smoothness(trail),steps=st)

def run(name, env, seeds):
    print(f"\n=== {name} ({len(seeds)} lượt) ===")
    print(f"{'Bộ lái':22s}{'Tới đích':>9s}{'Va chạm':>8s}{'e_y(m)':>8s}{'Đánh võng°':>12s}{'Bước':>7s}")
    for mode in ["Pure-Pursuit","ms_guided_smooth","ms_guided_caps","ms_guided_lipsnet"]:
        rs=[episode(env,sd,mode) for sd in seeds]
        arr=np.mean([r['arr'] for r in rs])*100; co=np.mean([r['coll'] for r in rs])*100
        ey=np.mean([r['ey'] for r in rs]); sm=np.mean([r['smooth'] for r in rs]); stp=np.mean([r['steps'] for r in rs])
        print(f"{mode:22s}{arr:>8.0f}%{co:>7.0f}%{ey:>8.3f}{sm:>12.2f}{stp:>7.0f}")

run("SET A — 1000-map", TrackEnv(config=dict(domain_random=False)), list(range(2000,2070)))
fixed=[]
for nm in ["apartment_a","apartment_b","apartment_c","test_c_curve","test_s_curve","test_u_turn"]:
    mm=build_map(nm); fixed.append({"grid":mm["grid"],"cell":mm["cell"],"origin":mm["origin"],"dock":mm["dock"],"points":mm["points"]})
sm=nd.load_saved_map(); fixed.append({"grid":sm["grid"],"cell":sm["cell"],"origin":sm["origin"],"dock":sm["dock"],"points":sm["points"]})
run("SET B — map cũ", TrackEnv(config=dict(domain_random=False, maps=fixed)), list(range(50)))
print("DONE")
