"""Builds map_apartment_complex_v1.ipynb — a STANDALONE map-config notebook.

Per the user's request: only the coordinate configuration of the new apartment
map + a visualization. NO navigation/algorithm code. The notebook prints the full
config dict (JSON), lists every coordinate group (bounds, corridor free-space,
isolated dock, destinations, resident patrol routes), and draws the map.
"""
import json
import os

NB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "map_apartment_complex_v1.ipynb")
cells = []
def md(s): cells.append(("markdown", s))
def code(s): cells.append(("code", s))

md("""# Map config — `apartment_complex_v1` (mới hoàn toàn)

Cấu hình toạ độ một **tầng hành lang chung cư lớn** để bạn tự đưa vào hệ thống
test điều hướng. Notebook này **chỉ chứa dữ liệu toạ độ + hình minh hoạ**, không
có code thuật toán.

**Địa hình:**
- Hành lang chính hình **chữ T**, rộng **2.0 m** (5 ô × 0.40 m).
- Nhiều **ngách thẳng vuông góc** rẽ vào cửa căn hộ.
- **Hành lang vòng cung** (2 cung nối thành vòng) **bao quanh giếng trời** ở giữa.
- **Dock biệt lập** (phòng kỹ thuật) — tường kín, **chỉ 1 cửa hẹp** ra hành lang.
- **4 điểm đến** (cửa căn hộ) nằm sâu trong ngách thẳng + trên vòng cung, xa dock.
- **3 quỹ đạo tuần tra của cư dân** (waypoint loops) dọc hành lang.

Hệ quy chiếu: lưới `rows×cols`, mỗi ô `cell_size_m`; tâm ô (r,c) trong toạ độ
thế giới = `(origin_x+(c+0.5)*cell, origin_y+(r+0.5)*cell)`.""")

md("## 1. Sinh & in cấu hình toạ độ đầy đủ (JSON)")
code("""import os, sys, json
sys.path.insert(0, os.path.abspath("."))
from apartment_complex_map import export_config
CONFIG = export_config()            # also returns the dict
print(json.dumps(CONFIG, indent=2, ensure_ascii=False))""")

md("## 2. Các nhóm toạ độ (tách riêng cho dễ dùng)")
code("""print("== Giới hạn map (world) ==");            print(CONFIG["map_bounds_world"])
print("\\n== Lưới ==");                             print(CONFIG["grid"])
print("\\n== Hành lang (ô đi được) ==")
print("  spine ngang:", CONFIG["corridors_cells"]["horizontal_spine"])
print("  nhánh dọc  :", CONFIG["corridors_cells"]["vertical_branch"])
print("  rộng (m)   :", CONFIG["corridors_cells"]["width_m"])
print("  ngách spine:", CONFIG["corridors_cells"]["apartment_niches_off_spine"])
print("  ngách nhánh:", CONFIG["corridors_cells"]["apartment_niches_off_branch"])
print("  vòng cung quanh giếng trời:", CONFIG["corridors_cells"]["curved_ring_around_lightwell"])
print("\\n== Dock biệt lập ==");                    print(CONFIG["dock_isolated"])
print("\\n== Điểm đến (cửa căn hộ) ==")
for k, v in CONFIG["destinations"].items(): print(f"  D{k}: {v}")
print("\\n== Quỹ đạo cư dân (patrol) ==")
for i, r in enumerate(CONFIG["resident_patrols"]):
    print(f"  resident {i+1}: cells={r['waypoints_cells']}")
    print(f"             world={[ [round(x,2),round(y,2)] for x,y in r['waypoints_world'] ]}")""")

md("## 3. Toạ độ KHÔNG GIAN HÀNH LANG (mọi ô robot đi được)")
code("""from apartment_complex_map import build
import numpy as np
M = build(); g = M["grid"]
free_cells = [(int(r), int(c)) for r, c in zip(*np.where(g == 1))]
print(f"số ô đi được (free cells): {len(free_cells)}")
print("ví dụ 20 ô đầu (r,c):", free_cells[:20])
# world-coord list also available:
ox, oy = M["origin"]; cell = M["cell"]
free_world = [(round(ox+(c+0.5)*cell,2), round(oy+(r+0.5)*cell,2)) for r,c in free_cells]
print("ví dụ 5 toạ độ world:", free_world[:5])""")

md("## 4. Hình minh hoạ map")
code("""import matplotlib.pyplot as plt, numpy as np
w = M["world"]
fig, ax = plt.subplots(figsize=(12, 10))
# free space (light) + walls (segments)
ox, oy = M["origin"]; cell = M["cell"]
for (r, c) in free_cells:
    ax.add_patch(plt.Rectangle((ox+c*cell, oy+r*cell), cell, cell,
                               facecolor="#eef3f7", edgecolor="none"))
for (x1, y1, x2, y2) in w.segments:
    ax.plot([x1, x2], [y1, y2], color="#333", lw=1.0)
# dock, destinations, patrols
ax.plot(*M["dock"], "s", color="#2a8f2a", ms=16, label="DOCK (biệt lập)")
for k, xy in M["points"].items():
    ax.plot(*xy, "o", color="#d22", ms=12)
    ax.annotate(f"D{k}", xy, textcoords="offset points", xytext=(0, 9),
                ha="center", weight="bold", color="#d22")
colors = ["#e8902a", "#9b59b6", "#1f9e6e"]
for i, route in enumerate(M["ped_routes"]):
    a = np.array(route)
    ax.plot(a[:, 0], a[:, 1], "--o", color=colors[i % 3], lw=1.6, ms=5,
            label=f"cư dân {i+1} (patrol)")
ax.set_aspect("equal"); ax.legend(loc="lower right", fontsize=8)
ax.set_title("apartment_complex_v1 — hành lang chữ T + ngách + vòng cung quanh giếng trời + dock biệt lập")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
plt.tight_layout(); plt.show()""")

md("""## 5. Ghi chú dùng dữ liệu

- **Đi được** ⇔ ô có giá trị `1` trong `grid` (mục 3) hoặc điểm nằm trong các vùng
  hành lang/ngách/vòng cung ở mục 2.
- **Dock biệt lập**: chỉ vào/ra qua 1 ô cửa `single_door_cell` + `narrow_stub`.
  Chặn ô cửa đó là dock tách hẳn khỏi mọi điểm đến (đã kiểm chứng).
- **Điểm đến**: `world_xy` để nạp thẳng làm goal; `cell` nếu bạn test trên lưới.
- **Quỹ đạo cư dân**: mỗi `waypoints_world` là 1 vòng lặp — cho người đi qua lại
  giữa các waypoint để ép test né vật cản động.
- File `apartment_complex_v1.json` (cùng thư mục) chứa đúng config này để nạp ngoài.""")

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
print("wrote", NB, "cells:", len(cells))
