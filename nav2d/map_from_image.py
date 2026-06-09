"""Chuyển ẢNH floor-plan (.png/.jpg) -> Occupancy Grid (hành lang robot đi được).

Ý tưởng: trong bản vẽ kiến trúc, HÀNH LANG thường là vùng MÀU XÁM đồng nhất, còn
phòng = trắng/kem (+ nội thất), tường = nét đen, căn hộ tô màu (theo chú thích).
=> tách "pixel xám" (độ bão hoà thấp + độ sáng trung bình) làm vùng đi được, rồi
hạ mẫu xuống lưới ô, và giữ THÀNH PHẦN LIÊN THÔNG LỚN NHẤT (bỏ đốm xám lạc ở chú
thích/logo/chữ).

Hàm chính:
  image_to_grid(path, ...) -> dict(grid, cell, origin)   # 1 = free corridor
  save_grid_npy(path_img, out_npy, ...)                   # lưu .npy + preview

Lưu ý trung thực: đây là image-segmentation theo MÀU, không phải SLAM. Bản vẽ thật
có thể cần chỉnh `sat_max / val_lo / val_hi / free_frac` cho khớp tông xám, và đôi
khi phải dọn tay vài chỗ. Cửa căn hộ (điểm giao) KHÔNG tự nhận ra -> đánh dấu tay
trong GUI.
"""

from __future__ import annotations

import argparse
from collections import deque

import numpy as np


def _largest_component(free: np.ndarray) -> np.ndarray:
    """Giữ vùng free liên thông lớn nhất (loại đốm xám lạc)."""
    R, C = free.shape
    seen = np.zeros_like(free, dtype=bool)
    best = None; best_sz = 0
    for r in range(R):
        for c in range(C):
            if free[r, c] and not seen[r, c]:
                comp = []; q = deque([(r, c)]); seen[r, c] = True
                while q:
                    a = q.popleft(); comp.append(a)
                    for nr, nc in ((a[0]+1,a[1]),(a[0]-1,a[1]),(a[0],a[1]+1),(a[0],a[1]-1)):
                        if 0 <= nr < R and 0 <= nc < C and free[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True; q.append((nr, nc))
                if len(comp) > best_sz:
                    best_sz = len(comp); best = comp
    out = np.zeros_like(free, dtype=np.uint8)
    if best:
        for (r, c) in best:
            out[r, c] = 1
    return out


def image_to_grid(path, target_cols=120, cell=0.4,
                  sat_max=28, val_lo=70, val_hi=205, free_frac=0.35,
                  keep_largest=True):
    """Trả {grid, cell, origin}. grid uint8: 1=hành lang đi được, 0=tường/phòng."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    a = np.asarray(img, dtype=np.int16)                       # (H,W,3)
    H, W = a.shape[:2]
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = mx - mn                                             # ~ độ bão hoà
    val = (r + g + b) / 3.0                                   # độ sáng
    corridor = (sat <= sat_max) & (val >= val_lo) & (val <= val_hi)   # pixel xám

    # hạ mẫu xuống lưới ô
    cols = max(20, int(target_cols)); scale = W / cols
    rows = max(1, int(round(H / scale)))
    grid = np.zeros((rows, cols), dtype=np.uint8)
    ys = (np.linspace(0, H, rows + 1)).astype(int)
    xs = (np.linspace(0, W, cols + 1)).astype(int)
    for i in range(rows):
        for j in range(cols):
            block = corridor[ys[i]:ys[i+1], xs[j]:xs[j+1]]
            if block.size and block.mean() >= free_frac:
                grid[i, j] = 1
    # ảnh: y xuống dưới; lưới world: y lên trên -> lật để khớp toạ độ
    grid = np.flipud(grid)
    if keep_largest:
        grid = _largest_component(grid)
    R, Cc = grid.shape
    origin = (-Cc * cell / 2.0, -R * cell / 2.0)
    return {"grid": grid, "cell": float(cell), "origin": origin,
            "free_cells": int(grid.sum()), "shape": (R, Cc)}


def save_grid_npy(path_img, out_npy="floorplan_grid.npy", preview="floorplan_preview.png", **kw):
    res = image_to_grid(path_img, **kw)
    np.save(out_npy, res["grid"])
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        g = res["grid"]
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.imshow(g, origin="lower", cmap="Greys", vmin=0, vmax=1)
        ax.set_title(f"Hành lang trích từ ảnh — free cells {res['free_cells']} | {res['shape']}")
        ax.set_xticks([]); ax.set_yticks([])
        fig.savefig(preview, dpi=90, bbox_inches="tight"); plt.close(fig)
    except Exception:
        preview = None
    res["out_npy"] = out_npy; res["preview"] = preview
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--out", default="floorplan_grid.npy")
    ap.add_argument("--cols", type=int, default=120)
    ap.add_argument("--cell", type=float, default=0.4)
    ap.add_argument("--sat-max", type=int, default=28)
    ap.add_argument("--val-lo", type=int, default=70)
    ap.add_argument("--val-hi", type=int, default=205)
    ap.add_argument("--free-frac", type=float, default=0.35)
    a = ap.parse_args()
    res = save_grid_npy(a.image, a.out, cols=a.cols, cell=a.cell, sat_max=a.sat_max,
                        val_lo=a.val_lo, val_hi=a.val_hi, free_frac=a.free_frac)
    print(f"grid {res['shape']} | free cells {res['free_cells']} -> {res['out_npy']}")
    if res["preview"]:
        print("preview:", res["preview"])
