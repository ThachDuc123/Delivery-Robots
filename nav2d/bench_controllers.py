"""Benchmark 3 bộ lái (Pure-pursuit, RL v1, RL v2) trên TẤT CẢ map đã test.
Đo: giao đủ điểm?, độ đánh võng (deg/step), số bước (thời gian), về dock.
Không có người đi bộ -> cô lập so sánh bộ lái.
"""
import sys, math, time
sys.path.insert(0, ".")
import numpy as np
import nav_demo as nd
import hybrid_controller as hc
from hybrid_nav import RLTracker, MAX_TURN
from fixed_maps import build_map

DT = 0.1

def smoothness(trail):
    tr = np.array(trail); d = np.diff(tr, axis=0)
    n = np.linalg.norm(d, axis=1); d = d[n > 1e-4]
    if len(d) < 2: return 0.0
    ang = np.arctan2(d[:, 1], d[:, 0])
    return float(np.degrees(np.abs((np.diff(ang) + math.pi) % (2*math.pi) - math.pi).mean()))

def get_map(name):
    if name == "apartment_complex":
        sm = nd.load_saved_map()
        return sm["grid"], sm["cell"], sm["origin"], sm["dock"], sm["points"]
    m = build_map(name)
    return m["grid"], m["cell"], m["origin"], m["dock"], m["points"]

MAPS = ["apartment_a", "apartment_b", "apartment_c",
        "test_c_curve", "test_s_curve", "test_u_turn", "apartment_complex"]

# load RL trackers once, reuse (reset state per run)
TRK = {"RL v1": RLTracker("ms_mixed_robust", lookahead=0.7),
       "RL v2": RLTracker("ms_mixed_robust_v2", lookahead=0.7)}

def make_policy(tag):
    tr = TRK[tag]
    def pol(world, pos, heading, look):
        a = tr.action(world, pos, heading, look)
        return float(a[0]), float(a[1]) * MAX_TURN
    return tr, pol

def run(name, ctrl, max_steps=15000):
    g, cell, origin, dock, pts = get_map(name)
    picks = list(pts)                       # giao TẤT CẢ điểm của map
    local = None
    if ctrl != "Pure-pursuit":
        tr, local = make_policy(ctrl); tr.reset()
    mc = hc.MissionController(g, cell, origin, dock, pts, log_fn=lambda s: None, local_policy=local)
    t0 = time.time()
    r = mc.run(picks, peds=None, max_steps=max_steps)
    tr_arr = np.array(r["trail"])
    visited = sum(1 for k in picks if np.linalg.norm(tr_arr - np.array(pts[k]), axis=1).min() < 0.5)
    steps = max(len(r["trail"]) - 1, 0)     # 1 entry/bước (run() không trả 'steps')
    return dict(n=len(picks), visited=visited, deliv=len(r["delivered"]),
                ret=r["returned_dock"], smooth=smoothness(r["trail"]),
                steps=steps, sec=steps*DT)

if __name__ == "__main__":
    CTRLS = ["Pure-pursuit", "RL v1", "RL v2"]
    rows = []
    print(f"{'Map':16s}{'Ctrl':14s}{'Giao':>8s}{'Về dock':>9s}{'Võng(°)':>9s}{'Bước':>7s}{'Giây':>7s}", flush=True)
    print("-"*72, flush=True)
    agg = {c: dict(vis=0, tot=0, smooth=[], steps=[], ret=0, nmap=0) for c in CTRLS}
    for name in MAPS:
        for c in CTRLS:
            try:
                d = run(name, c)
                rows.append((name, c, d))
                print(f"{name:16s}{c:14s}{d['visited']}/{d['n']:>6d}{str(d['ret']):>9s}"
                      f"{d['smooth']:>9.2f}{d['steps']:>7d}{d['sec']:>7.1f}", flush=True)
                a = agg[c]; a['vis']+=d['visited']; a['tot']+=d['n']; a['smooth'].append(d['smooth'])
                a['steps'].append(d['steps']); a['ret']+=int(d['ret']); a['nmap']+=1
            except Exception as e:
                print(f"{name:16s}{c:14s}  ERROR {e}", flush=True)
        print("-"*72, flush=True)
    print("\n=== TRUNG BÌNH ===", flush=True)
    print(f"{'Ctrl':14s}{'Giao đủ':>10s}{'Về dock':>9s}{'Võng TB(°)':>12s}{'Bước TB':>9s}", flush=True)
    for c in CTRLS:
        a = agg[c]
        print(f"{c:14s}{a['vis']}/{a['tot']:>7d}{a['ret']}/{a['nmap']:>6d}"
              f"{np.mean(a['smooth']):>12.2f}{np.mean(a['steps']):>9.0f}", flush=True)
    # bar charts
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    x = np.arange(len(MAPS)); w = 0.25
    cols = {"Pure-pursuit": "#1f77b4", "RL v1": "#d62728", "RL v2": "#2ca02c"}
    for i, c in enumerate(CTRLS):
        sm_ = [next((d['smooth'] for n,cc,d in rows if n==M and cc==c), 0) for M in MAPS]
        st_ = [next((d['steps'] for n,cc,d in rows if n==M and cc==c), 0) for M in MAPS]
        su_ = [next((d['visited']/max(d['n'],1)*100 for n,cc,d in rows if n==M and cc==c), 0) for M in MAPS]
        ax[0].bar(x+(i-1)*w, sm_, w, label=c, color=cols[c])
        ax[1].bar(x+(i-1)*w, st_, w, label=c, color=cols[c])
        ax[2].bar(x+(i-1)*w, su_, w, label=c, color=cols[c])
    ax[0].set_title("Độ đánh võng (°/bước) — thấp=mượt"); ax[1].set_title("Số bước (thời gian) — thấp=nhanh")
    ax[2].set_title("Tỉ lệ giao đủ điểm (%)")
    for a in ax:
        a.set_xticks(x); a.set_xticklabels([m.replace("apartment_","apt_") for m in MAPS], rotation=40, ha="right", fontsize=7)
        a.legend(fontsize=7); a.grid(alpha=.3, axis="y")
    plt.tight_layout(); plt.savefig("compare_controllers.png", dpi=100, bbox_inches="tight")
    print("\nsaved compare_controllers.png")
