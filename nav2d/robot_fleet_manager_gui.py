"""BƯỚC 6 — Robot Fleet Manager (Desktop GUI, Tkinter + Matplotlib).

Đóng gói toàn bộ kiến trúc Hybrid (SLAM map + A* inflate + Mission Controller +
Safety Shield + Dynamic Replanning) vào 1 app điều khiển:

  [Upload Map]        chọn file .npy (occupancy grid) hoặc dùng map SLAM mặc định.
  [Set Dock]          bật chế độ click -> click lên map chốt Dock (🏠).
  [Đăng ký giao hàng] bật chế độ click -> click nhiều điểm trạm (chấm xanh dương).
  [Gen Map cho Robot] chạy Inflation (vùng an toàn) trên map -> hiện overlay.
  Checklist           tick các trạm sẽ đi + [Xác nhận điểm giao hàng].
  Pedestrian          slider số người (0-10) + dropdown Random/Patrol.
  [CHẠY GIAO HÀNG]    chạy Hybrid Controller, vẽ robot real-time + bắn log.
  Log browser         🟢 giao xong / 🔴 bỏ qua trạm hẹp / ⚠️ lách người / ✅ về dock.

Chạy:  .venv\\Scripts\\python.exe robot_fleet_manager_gui.py
"""

from __future__ import annotations

import os
import queue
import sys
import threading

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


class FleetManagerApp:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk
        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.tk = tk; self.ttk = ttk
        self.root = root
        root.title("Robot Fleet Manager — Hybrid Navigation")

        # ---- state ----
        self.grid = None; self.cell = 0.4; self.origin = (0.0, 0.0)
        self.inflated = None
        self.raw_image = None; self.raw_image_path = None   # ảnh upload chờ Gen Map
        self.dock = None
        self.registered = []                 # list of (x,y)
        self.click_mode = None               # 'dock' | 'register' | None
        self.pick_vars = []                  # checkbutton vars
        self.confirmed_picks = []
        self.log_q = queue.Queue()
        self.running = False

        # ---- layout: left canvas, right control panel ----
        left = tk.Frame(root); left.pack(side="left", fill="both", expand=True)
        right = tk.Frame(root, width=320); right.pack(side="right", fill="y")

        self.fig = Figure(figsize=(7, 6)); self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_click)

        self._panel(right)
        self._load_default_map()

    # ----------------------------- panel ---------------------------------- #
    def _panel(self, p):
        tk, ttk = self.tk, self.ttk
        def section(txt):
            ttk.Label(p, text=txt, font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(8, 1), padx=6)
        section("1. Bản đồ")
        ttk.Button(p, text="Upload Map (.npy/ảnh)", command=self._upload).pack(fill="x", padx=6)
        ttk.Button(p, text="Gen Map cho Robot (Inflation)", command=self._gen_map).pack(fill="x", padx=6, pady=1)
        ttk.Button(p, text="💾 Lưu Map + điểm (.npz)", command=self._save_scene).pack(fill="x", padx=6)
        ttk.Button(p, text="📂 Nạp Map đã lưu", command=self._load_scene).pack(fill="x", padx=6, pady=1)
        section("2. Điểm")
        ttk.Button(p, text="Set Dock 🏠", command=lambda: self._set_mode("dock")).pack(fill="x", padx=6)
        ttk.Button(p, text="Đăng ký Tọa độ Giao hàng", command=lambda: self._set_mode("register")).pack(fill="x", padx=6, pady=1)
        ttk.Button(p, text="Xóa điểm đăng ký", command=self._clear_pts).pack(fill="x", padx=6)
        section("3. Chọn trạm giao (tick các điểm đã đăng ký)")
        self.checkframe = tk.Frame(p); self.checkframe.pack(fill="x", padx=6)
        ttk.Button(p, text="Xác nhận điểm giao hàng", command=self._confirm).pack(fill="x", padx=6, pady=1)
        section("4. Người đi bộ")
        self.nped = tk.IntVar(value=3)
        tk.Scale(p, from_=0, to=10, orient="horizontal", variable=self.nped,
                 label="Số người").pack(fill="x", padx=6)
        self.behavior = tk.StringVar(value="Random Walk")
        ttk.Combobox(p, textvariable=self.behavior, state="readonly",
                     values=["Random Walk", "Patrol"]).pack(fill="x", padx=6)
        section("5. Bộ lái cục bộ")
        self.driver = tk.StringVar(value="Pure-pursuit (mượt nhất)")
        ttk.Combobox(p, textvariable=self.driver, state="readonly",
                     values=["Pure-pursuit (mượt nhất)",
                             "RL mượt (ms_guided_smooth)",
                             "RL mượt v2 (ms_guided_smooth2)",
                             "RL v2 (ms_mixed_robust_v2)",
                             "RL v1 (ms_mixed_robust)"]).pack(fill="x", padx=6)
        section("6. Thực thi")
        self.start_btn = tk.Button(p, text="▶  CHẠY GIAO HÀNG", bg="#2a8f2a", fg="white",
                                   font=("Segoe UI", 11, "bold"), command=self._start)
        self.start_btn.pack(fill="x", padx=6, pady=4)
        section("Log")
        self.logbox = tk.Text(p, height=14, width=40, wrap="word", font=("Consolas", 8))
        self.logbox.pack(fill="both", expand=True, padx=6, pady=(0, 6))

    # ----------------------------- map ------------------------------------ #
    def _load_default_map(self):
        try:
            import nav_demo as nd
            sm = nd.load_saved_map()
            self.grid = sm["grid"]; self.cell = sm["cell"]; self.origin = sm["origin"]
            self.dock = tuple(sm["dock"])
            self.registered = [tuple(v) for v in sm["points"].values()]
            self._log("Đã nạp map SLAM mặc định (generated_apartment_map).")
        except Exception as e:
            self._log(f"Chưa có map mặc định ({e}). Hãy Upload Map.")
        self._build_checklist()      # các điểm sẵn có hiện ngay ở mục 'Chọn trạm giao'
        self._redraw()

    def _upload(self):
        from tkinter import filedialog
        fn = filedialog.askopenfilename(filetypes=[
            ("Ảnh / Lưới", "*.png *.jpg *.jpeg *.bmp *.npy"),
            ("Ảnh floor-plan", "*.png *.jpg *.jpeg *.bmp"),
            ("Occupancy grid", "*.npy"), ("All", "*.*")])
        if not fn:
            return
        ext = os.path.splitext(fn)[1].lower()
        self.dock = None; self.registered = []; self.confirmed_picks = []
        self.inflated = None; self._build_checklist()
        if ext == ".npy":
            g = np.load(fn)
            self.grid = (g > 0).astype(np.uint8) if g.ndim == 2 else g
            self.raw_image = None; self.raw_image_path = None
            R, C = self.grid.shape
            self.origin = (-C*self.cell/2.0, -R*self.cell/2.0)
            self._log(f"Đã upload lưới {os.path.basename(fn)} ({R}x{C}). Có thể chạy luôn.")
        else:
            from PIL import Image
            self.raw_image = np.asarray(Image.open(fn).convert("RGB"))
            self.raw_image_path = fn; self.grid = None
            self._log(f"Đã upload ẢNH {os.path.basename(fn)}. Bấm 'Gen Map cho Robot' "
                      f"để trích hành lang -> map (ảnh gốc sẽ được bỏ).")
        self._redraw()

    def _save_scene(self):
        """Lưu CẢ map + dock + điểm giao ra .npz để lần sau nạp chạy luôn."""
        if self.grid is None:
            self._log("Chưa có map để lưu."); return
        from tkinter import filedialog
        fn = filedialog.asksaveasfilename(defaultextension=".npz",
                                          initialfile="saved_scene.npz",
                                          filetypes=[("Scene", "*.npz")])
        if not fn:
            return
        np.savez(fn, grid=self.grid, cell=self.cell, origin=np.array(self.origin),
                 dock=np.array(self.dock if self.dock is not None else [0, 0]),
                 has_dock=np.array(self.dock is not None),
                 points=np.array(self.registered, dtype=float) if self.registered else np.zeros((0, 2)))
        self._log(f"💾 Đã lưu scene -> {os.path.basename(fn)} "
                  f"(map + dock + {len(self.registered)} điểm). Lần sau bấm 'Nạp Map đã lưu'.")

    def _load_scene(self):
        """Nạp lại map + dock + điểm đã lưu -> chạy tiếp ngay."""
        from tkinter import filedialog
        fn = filedialog.askopenfilename(filetypes=[("Scene", "*.npz"), ("All", "*.*")])
        if not fn:
            return
        d = np.load(fn, allow_pickle=True)
        self.grid = d["grid"].astype(np.uint8); self.cell = float(d["cell"])
        self.origin = tuple(d["origin"]); self.raw_image = None; self.raw_image_path = None
        self.dock = tuple(d["dock"]) if bool(d["has_dock"]) else None
        self.registered = [tuple(p) for p in d["points"]]
        self.confirmed_picks = []; self.inflated = None
        self._build_checklist()
        self._log(f"📂 Đã nạp {os.path.basename(fn)}: map + dock + {len(self.registered)} điểm. "
                  f"Bấm 'Gen Map' rồi chọn trạm + CHẠY.")
        self._redraw()

    def _gen_map(self):
        # (1) Nếu đang có ẢNH upload -> trích hành lang thành lưới, rồi BỎ ảnh gốc.
        if self.raw_image is not None and self.raw_image_path:
            from map_from_image import image_to_grid
            try:
                res = image_to_grid(self.raw_image_path, target_cols=120, cell=self.cell)
            except Exception as e:
                self._log(f"Lỗi trích ảnh: {e}"); return
            self.grid = res["grid"]; self.origin = res["origin"]
            self.raw_image = None; self.raw_image_path = None        # xoá ảnh gốc
            self._log(f"🗺️ Đã gen map từ ảnh: {res['shape']}, hành lang {res['free_cells']} ô. "
                      f"(đã bỏ ảnh gốc, chỉ giữ map)")
            if int(self.grid.sum()) < 40:
                self._log("⚠️ Trích được rất ít hành lang — ảnh có thể cần chỉnh ngưỡng màu "
                          "(xám) hoặc crop bớt chú thích.")
        if self.grid is None:
            self._log("Chưa có map (upload ảnh hoặc .npy trước)."); return
        # (2) Inflation -> vùng an toàn
        from hybrid_controller import inflate_map
        self.inflated = inflate_map(self.grid, 1)
        self._log("✅ Gen Map xong: đã tính vùng an toàn (Inflation 0.3m).")
        self._build_checklist(); self._redraw()

    # ----------------------------- clicks --------------------------------- #
    def _set_mode(self, m):
        self.click_mode = m
        self._log(f"Chế độ click: {'Set Dock' if m=='dock' else 'Đăng ký điểm giao'} "
                  f"-> click lên bản đồ.")

    def _on_click(self, ev):
        if ev.inaxes != self.ax or ev.xdata is None or self.click_mode is None:
            return
        p = (float(ev.xdata), float(ev.ydata))
        if self.click_mode == "dock":
            self.dock = p; self._log(f"🏠 Dock = ({p[0]:.1f}, {p[1]:.1f})")
        else:
            self.registered.append(p)
            self._log(f"➕ Trạm {len(self.registered)-1} = ({p[0]:.1f}, {p[1]:.1f})")
            # điểm vừa đăng ký HIỆN NGAY xuống mục 'Chọn trạm giao' để tick
            self._build_checklist()
        self._redraw()

    def _clear_pts(self):
        self.registered = []; self.confirmed_picks = []
        self._build_checklist(); self._redraw(); self._log("Đã xóa điểm đăng ký.")

    # --------------------------- checklist -------------------------------- #
    def _build_checklist(self):
        for w in self.checkframe.winfo_children():
            w.destroy()
        self.pick_vars = []
        for i, _ in enumerate(self.registered):
            v = self.tk.IntVar(value=1)
            self.ttk.Checkbutton(self.checkframe, text=f"Trạm {i}", variable=v).pack(anchor="w")
            self.pick_vars.append(v)

    def _confirm(self):
        self.confirmed_picks = [i for i, v in enumerate(self.pick_vars) if v.get()]
        self._log(f"Xác nhận giao tới: {self.confirmed_picks}")
        self._redraw()

    # --------------------------- drawing ---------------------------------- #
    def _redraw(self, robot=None, peds=None, trail=None):
        ax = self.ax; ax.clear()
        if self.grid is None and self.raw_image is not None:
            # chưa Gen Map: hiện ảnh floor-plan vừa upload
            ax.imshow(self.raw_image)
            ax.set_title("Ảnh floor-plan đã upload — bấm 'Gen Map cho Robot'", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([]); self.canvas.draw_idle(); return
        if self.grid is not None:
            ox, oy = self.origin; cell = self.cell
            show = self.inflated if self.inflated is not None else self.grid
            free = np.argwhere(self.grid == 1)
            for (r, c) in free:
                ax.add_patch(__import__("matplotlib").patches.Rectangle(
                    (ox + c*cell, oy + r*cell), cell, cell, facecolor="#eef3f7", edgecolor="none"))
            if self.inflated is not None:
                infl = np.argwhere((self.grid == 1) & (self.inflated == 0))
                for (r, c) in infl:
                    ax.add_patch(__import__("matplotlib").patches.Rectangle(
                        (ox + c*cell, oy + r*cell), cell, cell, facecolor="#ffe0b0", edgecolor="none"))
            from hybrid_controller import MissionController
            w = MissionController._world_from_grid(self.grid, cell, self.origin)
            for (x1, y1, x2, y2) in w.segments:
                ax.plot([x1, x2], [y1, y2], color="#333", lw=0.6)
            R, C = self.grid.shape
            ax.set_xlim(ox - cell, ox + C*cell + cell); ax.set_ylim(oy - cell, oy + R*cell + cell)
        if self.dock is not None:
            ax.plot(*self.dock, "s", color="#2a8f2a", ms=14); ax.annotate("🏠", self.dock,
                    textcoords="offset points", xytext=(0, 8), ha="center")
        for i, p in enumerate(self.registered):
            sel = i in self.confirmed_picks
            ax.plot(*p, "o", color="#1f77b4" if not sel else "#d22", ms=9)
            ax.annotate(f"{i}", p, textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
        if trail is not None and len(trail) > 1:
            t = np.array(trail); ax.plot(t[:, 0], t[:, 1], "-", color="#2ca02c", lw=2)
        if robot is not None:
            ax.add_patch(__import__("matplotlib").patches.Circle(robot, 0.22, color="#1f3b73", zorder=9))
        if peds:
            for (px, py) in peds:
                ax.add_patch(__import__("matplotlib").patches.Circle((px, py), 0.28, color="#e8902a", zorder=8))
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        self.canvas.draw_idle()

    # --------------------------- run -------------------------------------- #
    def _log(self, msg):
        self.logbox.insert("end", msg + "\n"); self.logbox.see("end")

    def _start(self):
        if self.running:
            return
        if self.grid is None or self.dock is None or not self.confirmed_picks:
            self._log("Cần: map + Dock + xác nhận ≥1 trạm trước khi chạy."); return
        if self.inflated is None:
            self._gen_map()
        self.running = True; self.start_btn.config(state="disabled")
        self._log("▶ Bắt đầu chuyến giao hàng...")
        picks = list(self.confirmed_picks)
        points = {i: self.registered[i] for i in range(len(self.registered))}
        nped = int(self.nped.get()); behavior = self.behavior.get()
        d = self.driver.get()
        # khớp theo tên model trong nhãn (kiểm cái cụ thể hơn trước)
        rl_model = ("ms_guided_smooth2" if "ms_guided_smooth2" in d else
                    "ms_guided_smooth" if "ms_guided_smooth" in d else
                    "ms_mixed_robust_v2" if "ms_mixed_robust_v2" in d else
                    "ms_mixed_robust" if "ms_mixed_robust" in d else None)   # None = pure-pursuit
        t = threading.Thread(target=self._sim_thread, args=(picks, points, nped, behavior, rl_model), daemon=True)
        t.start()
        self.root.after(50, self._drain)

    def _sim_thread(self, picks, points, nped, behavior, rl_model=None):
        try:
            from hybrid_controller import MissionController
            from pedestrians2d import Pedestrians
            local_policy = None
            if rl_model and rl_model.startswith("ms_guided"):
                # Guided-RL mượt (smooth / smooth2): GuidedTracker tự là local_policy
                from hybrid_nav import GuidedTracker
                self.log_q.put(("log", f"Đang nạp RL mượt {rl_model} làm bộ lái..."))
                local_policy = GuidedTracker(model_name=rl_model)
            elif rl_model:
                # cắm model RL (v1/v2) làm bộ lái cục bộ (tuỳ chọn / fallback)
                from hybrid_nav import RLTracker, MAX_TURN
                self.log_q.put(("log", f"Đang nạp RL {rl_model} làm bộ lái..."))
                _tr = RLTracker(model_name=rl_model, lookahead=0.7)
                def local_policy(world, pos, heading, look):
                    a = _tr.action(world, pos, heading, look)
                    return float(a[0]), float(a[1]) * MAX_TURN
            mc = MissionController(self.grid, self.cell, self.origin, self.dock, points,
                                   log_fn=lambda s: self.log_q.put(("log", s)),
                                   local_policy=local_policy)
            peds = None
            if nped > 0:
                peds = Pedestrians(mc.world, np.random.default_rng(7), n=nped,
                                   speed_range=(0.5, 1.0), dt=0.1,
                                   grid_map={"grid": self.grid, "cell": self.cell, "origin": self.origin})
            res = mc.run(picks, peds=peds, max_steps=25000)
            self.log_q.put(("result", res))
        except Exception as e:
            self.log_q.put(("log", f"Lỗi: {e}"))
            self.log_q.put(("result", None))

    def _drain(self):
        try:
            while True:
                kind, payload = self.log_q.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "result":
                    if payload is not None:
                        self._animate(payload)
                    else:
                        self.running = False; self.start_btn.config(state="normal")
                    return
        except queue.Empty:
            pass
        self.root.after(50, self._drain)

    def _animate(self, res):
        trail = res["trail"]; pedtrail = res.get("pedtrail") or []
        step = max(1, len(trail) // 300)
        self._anim_idx = 0
        def tick():
            k = self._anim_idx
            if k >= len(trail):
                d = len(res["delivered"]); self._log(
                    f"— KẾT THÚC: giao {d} trạm, bỏ {len(res['cancelled'])}, "
                    f"về dock={res['returned_dock']}, va chạm người={res['ped_hits']} —")
                self.running = False; self.start_btn.config(state="normal"); return
            peds = pedtrail[k] if k < len(pedtrail) else None
            self._redraw(robot=trail[k], peds=peds, trail=trail[:k+1])
            self._anim_idx += step
            self.root.after(20, tick)
        tick()


def main():
    import tkinter as tk
    root = tk.Tk(); root.geometry("1180x720")
    FleetManagerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
