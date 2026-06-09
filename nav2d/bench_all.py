"""Benchmark THỐNG NHẤT tất cả bộ lái trong CÙNG khung (MissionController, 7 map):
Pure-pursuit, RL v1, RL v2, RL guided smooth, RL guided smooth2.
Đo: giao đủ điểm?, đánh võng (deg/step), số bước. Không người -> so thuần bộ lái.
"""
import sys, math, time
sys.path.insert(0, ".")
import numpy as np
import nav_demo as nd
import hybrid_controller as hc
from hybrid_nav import RLTracker, GuidedTracker, MAX_TURN
from fixed_maps import build_map

def smoothness(trail):
    tr=np.array(trail); d=np.diff(tr,axis=0); n=np.linalg.norm(d,axis=1); d=d[n>1e-4]
    if len(d)<2: return 0.0
    ang=np.arctan2(d[:,1],d[:,0])
    return float(np.degrees(np.abs((np.diff(ang)+math.pi)%(2*math.pi)-math.pi).mean()))

def get_map(name):
    if name=="apartment_complex":
        sm=nd.load_saved_map(); return sm["grid"],sm["cell"],sm["origin"],sm["dock"],sm["points"]
    m=build_map(name); return m["grid"],m["cell"],m["origin"],m["dock"],m["points"]

MAPS=["apartment_a","apartment_b","apartment_c","test_c_curve","test_s_curve","test_u_turn","apartment_complex"]

# tạo policy 1 lần, tái dùng
RL1=RLTracker("ms_mixed_robust",0.7); RL2=RLTracker("ms_mixed_robust_v2",0.7)
GS=GuidedTracker("ms_guided_smooth"); GS2=GuidedTracker("ms_guided_smooth2")
def wrap(tr):
    def pol(world,pos,heading,look):
        a=tr.action(world,pos,heading,look); return float(a[0]), float(a[1])*MAX_TURN
    return pol

def local_for(ctrl):
    if ctrl=="Pure-pursuit": return None
    if ctrl=="RL v1": RL1.reset(); return wrap(RL1)
    if ctrl=="RL v2": RL2.reset(); return wrap(RL2)
    if ctrl=="RL guided smooth": return GS
    if ctrl=="RL guided smooth2": return GS2

def run(name, ctrl, max_steps=15000):
    g,cell,origin,dock,pts=get_map(name); picks=list(pts)
    mc=hc.MissionController(g,cell,origin,dock,pts,log_fn=lambda s:None, local_policy=local_for(ctrl))
    r=mc.run(picks, peds=None, max_steps=max_steps)
    tr=np.array(r["trail"]); vis=sum(1 for k in picks if np.linalg.norm(tr-np.array(pts[k]),axis=1).min()<0.5)
    return dict(n=len(picks),vis=vis,ret=r["returned_dock"],smooth=smoothness(r["trail"]),steps=max(len(r["trail"])-1,0))

CTRLS=["Pure-pursuit","RL v1","RL v2","RL guided smooth","RL guided smooth2"]
rows={}; t0=time.time()
print(f"{'Map':16s}{'Ctrl':20s}{'Giao':>8s}{'Võng°':>9s}{'Bước':>7s}", flush=True)
print("-"*62, flush=True)
for nm in MAPS:
    for c in CTRLS:
        try:
            d=run(nm,c); rows[(nm,c)]=d
            print(f"{nm:16s}{c:20s}{d['vis']}/{d['n']:>5d}{d['smooth']:>9.2f}{d['steps']:>7d}", flush=True)
        except Exception as e:
            print(f"{nm:16s}{c:20s}  ERR {e}", flush=True)
    print("-"*62, flush=True)

print("\n=== TRUNG BÌNH (7 map) ===", flush=True)
print(f"{'Ctrl':20s}{'Giao đủ':>10s}{'Võng TB°':>10s}{'Bước TB':>9s}", flush=True)
for c in CTRLS:
    ds=[rows[(nm,c)] for nm in MAPS if (nm,c) in rows]
    vis=sum(d['vis'] for d in ds); tot=sum(d['n'] for d in ds)
    print(f"{c:20s}{vis}/{tot:>7d}{np.mean([d['smooth'] for d in ds]):>10.2f}{np.mean([d['steps'] for d in ds]):>9.0f}", flush=True)
print(f"(xong trong {time.time()-t0:.0f}s)")
