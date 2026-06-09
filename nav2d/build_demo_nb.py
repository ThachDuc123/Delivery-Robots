"""Builds delivery_demo.ipynb — the 3-phase delivery demo on apartment_complex_v1.

Phase 1: blind robot SLAM-maps the floor and SAVES the map.
Phase 2: load the saved map, PICK delivery points, robot routes (TSP+A*) and
         delivers them all then returns to dock.
Phase 3: residents walk the corridors; the robot drives its saved map and uses
         its sensors to dodge people SMOOTHLY (small radius, slow, no weaving,
         straightens back out afterwards).
Runs sequentially; GIFs are shown inline.
"""
import json, os
NB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "delivery_demo.ipynb")
cells = []
def md(s): cells.append(("markdown", s))
def code(s): cells.append(("code", s))

md("""# Robot giao hàng — Demo 3 giai đoạn (map `apartment_complex_v1`)

1. **Robot mù** → tự quét dựng bản đồ (SLAM) → **lưu map**.
2. **Chọn điểm giao** → robot tự tính đường (TSP + A*) trên map đã lưu → giao đủ → về dock.
3. **Có người đi lại** → robot đi trên map đã gen, dùng **cảm biến né người mượt**
   (bán kính nhỏ, đi chậm khi né, không đánh võng, né xong đi thẳng lại).

Chạy tuần tự từ trên xuống.""")

md("## Phase 1 — Robot mù: tự quét & LƯU bản đồ (SLAM)")
code("""import os, sys
sys.path.insert(0, os.path.abspath("."))
import nav_demo as nd, render_demo as rd
from IPython.display import Image, display

P1 = nd.build_and_save_map(seed=0)
print(f"Đã quét xong. Độ phủ bản đồ: {P1['coverage']*100:.1f}%  | số bước quét: {P1['explore_steps']}")
print(f"Bản đồ đã LƯU -> {P1['save_path']}")
g1 = rd.gif_phase1_mapping(None, P1['frames'], P1['occ'], 'results/gifs/demo1_mapping.gif')
display(Image(filename=g1))""")

md("""## Phase 2 — Chọn điểm giao → robot tự đi giao & về dock

Đổi `PICKS` thành danh sách điểm muốn giao (id từ 0..7). Robot dùng **bản đồ đã
lưu ở Phase 1**, tự sắp thứ tự tối ưu (TSP) và vạch đường (A*), giao hết rồi về dock.""")
code("""SM = nd.load_saved_map()   # <-- bản đồ robot TỰ DỰNG ở Phase 1 (không dùng map gốc)
print("Các điểm giao có trên map:", sorted(SM['points']))

PICKS = [0, 3, 7]          # <- chọn điểm muốn giao tới
# CONTROLLER:
#   "rl"  = dùng MÔ HÌNH ĐÃ TRAIN (RecurrentPPO/LSTM ms_mixed) lái bằng cảm biến
#           (đúng kiến trúc Hybrid: A* vạch tuyến + RL lái). Zero-shot trên map này.
#   "geo" = bộ lái hình học A*+pure-pursuit (chắc chắn tới, để đối chiếu).
CONTROLLER = "rl"

def run_deliver(picks, n_peds=0, seed=0):
    if CONTROLLER == "rl":
        return nd.deliver_rl(picks, saved_map=SM, n_peds=n_peds, seed=seed)
    return nd.deliver(picks, saved_map=SM, n_peds=n_peds, seed=seed)

P2 = run_deliver(PICKS)
print(f"Controller: {P2['controller'] if 'controller' in P2 else 'geometric'}")
print(f"Thứ tự đi: DOCK -> " + " -> ".join(f"D{p}" for p in P2['order']) + " -> DOCK")
dn = P2['delivered'] if isinstance(P2['delivered'], int) else len(P2['delivered'])
print(f"Đã giao: {dn} điểm | Về tới dock: {P2['returned_dock']} | số bước: {P2['steps']}")
g2 = rd.gif_delivery(P2, 'results/gifs/demo2_delivery.gif',
                     f"Phase 2 — giao {PICKS} ({CONTROLLER}, map tự dựng)")
display(Image(filename=g2))""")

md("""## Phase 3 — Có người đi lại → robot né mượt bằng cảm biến

Cùng map + điểm giao, nhưng thêm **người di chuyển** trong hành lang. Robot phát
hiện người phía trước bằng cảm biến (cone hẹp), **đi chậm + lách một chút** (bán
kính né nhỏ để không đánh võng), nếu quá sát thì **dừng nhường**, người qua rồi
**đi thẳng tiếp** như cũ.""")
code("""PICKS3 = [2, 6]
N_PEOPLE = 3
P3 = run_deliver(PICKS3, n_peds=N_PEOPLE, seed=1)
dn = P3['delivered'] if isinstance(P3['delivered'], int) else len(P3['delivered'])
print(f"Controller: {P3.get('controller','geometric')}")
print(f"Đã giao: {dn} điểm | Về tới dock: {P3['returned_dock']} | Số lần va chạm người: {P3['ped_hits']}")
g3 = rd.gif_delivery(P3, 'results/gifs/demo3_dodge.gif',
                     f"Phase 3 — né người ({CONTROLLER}, va chạm={P3['ped_hits']})")
display(Image(filename=g3))""")

md("""## Ghi chú
- **Phase 1** robot mù tự quét → lưu map ra `runs/slam_apartment_complex.npz`.
  Phase 2/3 **dựng lại thế giới TỪ map đã lưu** (`world_from_saved_map`) rồi mới
  chạy — nên robot thật sự đi trên bản đồ NÓ tự tạo, không phải map gốc.
- **`CONTROLLER="rl"`**: dùng **mô hình đã train** (RecurrentPPO/LSTM `ms_mixed`,
  sensor-only) làm bộ lái local — đúng kiến trúc Hybrid (A* vạch tuyến + RL lái).
  Vì model train trên map procedural ngẫu nhiên, chạy ở đây là **zero-shot**:
  giao được phần lớn điểm, ngách khó có thể chưa hoàn hảo.
  Đổi `CONTROLLER="geo"` để xem bộ lái hình học (A*+pure-pursuit) tới chắc 100%.
- **Né người (Phase 3):** cảm biến cone hẹp + bán kính né nhỏ, giảm tốc khi né,
  low-pass offset → **không đánh võng**; quá sát thì dừng nhường rồi đi thẳng lại.
- Đổi `PICKS` để giao tới các căn hộ khác nhau; `CONTROLLER` để đổi bộ lái.""")

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
