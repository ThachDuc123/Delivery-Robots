"""Interactive delivery — click delivery points on the map, press a button, watch
the robot run. No extra installs (uses matplotlib's built-in TkAgg window).

Run:   .venv\\Scripts\\python.exe interactive_delivery.py

Controls:
  * Click a numbered apartment door (grey circle) -> toggle it as a delivery stop
    (turns red). Click again to deselect.
  * Click anywhere in a free corridor -> add a CUSTOM delivery point (snapped to
    the nearest reachable free cell).
  * Button "Chay Robot" -> robot plans (TSP+A*) and drives the route live.
  * Button "Nguoi: Bat/Tat" -> toggle moving pedestrians.
  * Button "Xoa chon" -> clear the selection.
"""

from __future__ import annotations

import math
import sys

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

import nav_demo as nd


def main():
    sm = nd.load_saved_map()
    g = sm["grid"]; cell = sm["cell"]; ox, oy = sm["origin"]
    dock = sm["dock"]; base_pts = dict(sm["points"])
    world = nd.world_from_saved_map(sm)
    free = [(int(r), int(c)) for r, c in zip(*np.where(g == 1))]
    xmin, ymin = ox, oy
    xmax = ox + g.shape[1] * cell; ymax = oy + g.shape[0] * cell

    # --- mutable state ---
    state = {"picks": [], "custom": {}, "next_id": 100, "peds_on": True,
             "running": False}

    fig, ax = plt.subplots(figsize=(11, 9))
    plt.subplots_adjust(bottom=0.13)

    def snap_free(x, y):
        """nearest free cell centre to a clicked world point."""
        best = None; bd = 1e9
        for (r, c) in free:
            wx = ox + (c + 0.5) * cell; wy = oy + (r + 0.5) * cell
            d = (wx - x) ** 2 + (wy - y) ** 2
            if d < bd:
                bd = d; best = (wx, wy)
        return best if bd < (1.2 ** 2) else None

    def all_points():
        p = dict(base_pts); p.update(state["custom"]); return p

    def draw_static(title="Click chọn điểm giao → bấm 'Chay Robot'"):
        ax.clear()
        for (r, c) in free:
            ax.add_patch(plt.Rectangle((ox + c*cell, oy + r*cell), cell, cell,
                                       facecolor="#eef3f7", edgecolor="none"))
        for (x1, y1, x2, y2) in world.segments:
            ax.plot([x1, x2], [y1, y2], color="#333", lw=0.7)
        ax.plot(*dock, "s", color="#2a8f2a", ms=15, zorder=5)
        ax.annotate("DOCK", dock, textcoords="offset points", xytext=(0, 9),
                    ha="center", weight="bold", color="#2a8f2a")
        for pid, xy in all_points().items():
            sel = pid in state["picks"]
            ax.plot(*xy, "o", color="#d22" if sel else "#aaa",
                    ms=13 if sel else 8, zorder=6)
            ax.annotate(f"D{pid}", xy, textcoords="offset points", xytext=(0, 8),
                        ha="center", weight="bold",
                        color="#d22" if sel else "#777")
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        order_txt = " → ".join(f"D{p}" for p in state["picks"]) or "(chưa chọn)"
        ax.set_title(f"{title}\nĐã chọn: {order_txt}   |   Người: "
                     f"{'BẬT' if state['peds_on'] else 'TẮT'}", fontsize=10)
        fig.canvas.draw_idle()

    def on_click(event):
        if state["running"] or event.inaxes != ax or event.xdata is None:
            return
        x, y = event.xdata, event.ydata
        # near an existing point? toggle it
        for pid, xy in all_points().items():
            if (xy[0] - x) ** 2 + (xy[1] - y) ** 2 < (0.5 ** 2):
                if pid in state["picks"]:
                    state["picks"].remove(pid)
                else:
                    state["picks"].append(pid)
                draw_static(); return
        # else add a custom point at the nearest free cell
        sp = snap_free(x, y)
        if sp is not None:
            pid = state["next_id"]; state["next_id"] += 1
            state["custom"][pid] = sp; state["picks"].append(pid)
            draw_static()

    def animate(result):
        trail = result["trail"]; pedtrail = result.get("pedtrail") or []
        import matplotlib.patches as mp
        step = max(1, len(trail) // 300)
        robot = mp.Circle(trail[0], 0.22, color="#1f3b73", zorder=9); ax.add_patch(robot)
        pcircles = []
        line, = ax.plot([], [], color="#1f77b4", lw=2, zorder=7)
        xs, ys = [], []
        for k in range(0, len(trail), step):
            xs.append(trail[k][0]); ys.append(trail[k][1]); line.set_data(xs, ys)
            robot.center = trail[k]
            for pc in pcircles: pc.remove()
            pcircles = []
            if pedtrail and k < len(pedtrail):
                for (px, py) in pedtrail[k]:
                    c = mp.Circle((px, py), 0.28, color="#e8902a", zorder=8)
                    ax.add_patch(c); pcircles.append(c)
            fig.canvas.draw_idle(); plt.pause(0.001)
        ax.set_title(f"XONG: giao {len(result['delivered'])}/{len(state['picks'])}"
                     f"  | về dock: {result['returned_dock']}"
                     f"  | va chạm người: {result['ped_hits']}", fontsize=10)
        fig.canvas.draw_idle()

    def on_run(_):
        if state["running"] or not state["picks"]:
            return
        state["running"] = True
        draw_static(title="Đang chạy...")
        sm2 = dict(sm); sm2["points"] = all_points()
        res = nd.deliver(list(state["picks"]), saved_map=sm2,
                         n_peds=3 if state["peds_on"] else 0, seed=2, max_steps=25000)
        if not res.get("reachable", True):
            ax.set_title("Không định tuyến được các điểm đã chọn.", fontsize=10)
            fig.canvas.draw_idle()
        else:
            animate(res)
        state["running"] = False

    def on_peds(_):
        state["peds_on"] = not state["peds_on"]; draw_static()

    def on_clear(_):
        state["picks"].clear(); state["custom"].clear(); state["next_id"] = 100
        draw_static()

    b_run = Button(plt.axes([0.13, 0.03, 0.18, 0.06]), "Chay Robot")
    b_ped = Button(plt.axes([0.34, 0.03, 0.18, 0.06]), "Nguoi: Bat/Tat")
    b_clr = Button(plt.axes([0.55, 0.03, 0.18, 0.06]), "Xoa chon")
    b_run.on_clicked(on_run); b_ped.on_clicked(on_peds); b_clr.on_clicked(on_clear)
    fig.canvas.mpl_connect("button_press_event", on_click)

    draw_static()
    plt.show()


if __name__ == "__main__":
    main()
